import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/fiducial"
data_path = f"{root}/data/solar/fiducial"

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the background folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help="Oscillation weighting backend. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}
sample_key = name.split("_")[0].lower()

info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
input_prefix = f"{info['PATH']}/data/{info['GEOMETRY']}/{info['VERSION']}/{info['NAME']}{name}"
required_file = f"{input_prefix}/Config/GEANT4Label.npy"
if not os.path.exists(required_file):
    rprint(
        "[yellow][WARNING][/yellow] Missing optional input sample "
        f"{name} for {config}. Expected file not found: {required_file}. Skipping."
    )
    raise SystemExit(0)

if not os.path.exists(f"{save_path}/{args.folder.lower()}"):
    os.makedirs(f"{save_path}/{args.folder.lower()}")

user_input = {
    "workflow": "SIGNIFICANCE",
    "weights": {
        "marley": [
            "SignalParticleWeight",
            "SignalParticleWeightb8",
            "SignalParticleWeighthep",
        ],
        "neutron": ["SignalParticleWeight"],
        "gamma": ["SignalParticleWeight"],
        "alpha": ["SignalParticleWeight"],
        "radiological": ["SignalParticleWeight"],
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
        "radiological": ["radiological"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
        "radiological": ["rgb(120, 94, 240)"],
    },
    "yzoom": {"marley": [0, 6], "neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8], "radiological": [0, 6]},
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
    debug=args.debug,
)

run = compute_reco_workflow(
    run,
    configs,
    params=(
        {
            "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
            "DEFAULT_SIGNAL_AZIMUTH": ["mean", "day", "night"],
            "PARTICLE_TYPE": "signal",
            "PARTICLE_WEIGHTING": "volume",
            "OSCILLATION_BACKEND": args.oscillation_backend,
        }
        if "marley" in args.name
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    rm_branches=False,
    workflow=user_input["workflow"],
    debug=args.debug,
)

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    for name, energy in product(configs[config], args.energy):
        plot_list = []
        fig = make_subplots(rows=1, cols=1, subplot_titles=([energy]))

        def fill_gaps(hist):
            """Transfer half the weight of each non-zero bin to an adjacent zero bin on its left,
            filling isolated zeros that arise from finite MC statistics in the tail."""
            hist = np.asarray(hist, dtype=float)
            right = hist[1:]
            left = hist[:-1]
            transfer = right * 0.5 * ((right > 0) & (left == 0))
            hist[:-1] += transfer
            hist[1:] -= transfer
            return hist

        for (
            this_fiducial_x,
            this_fiducial_y,
            this_fiducial_z,
            (weight, weight_labels, color),
        ) in track(
            product(
                np.arange(0.00, detector_x / 4, 20),
                np.arange(0.00, detector_y / 4, 20),
                np.arange(0.00, detector_z / 4, 20),
                zip(
                    user_input["weights"][sample_key],
                    user_input["weight_labels"][sample_key],
                    user_input["colors"][sample_key],
                ),
            ),
            total=(3 if "marley" in args.name else 1)
            * len(np.arange(0.00, detector_x / 4, 20))
            * len(np.arange(0.00, detector_y / 4, 20))
            * len(np.arange(0.00, detector_z / 4, 20)),
            description=f"Iterating over cut configurations for {energy}...",
        ):
            # Quality cuts must always be applied so that the fid=(0,0,0) baseline
            # is comparable to fid>0 points. Previously fid=(0,0,0) used
            # mask=ones, bypassing OpFlash and surface requirements and inflating
            # its significance over every non-zero fiducial value.
            quality_mask = (
                (
                    run["Reco"]["SignalParticleSurface"] >= 0
                    if is_surface_background(str(root), sample_key)
                    else np.ones(
                        len(run["Reco"]["SignalParticleSurface"]), dtype=bool
                    )
                )
                * (
                    (run["Reco"]["SignalParticleSurface"] < 3)
                    if (
                        args.folder in ["Reduced", "Truncated"]
                        and is_surface_background(str(root), sample_key)
                    )
                    else np.ones(
                        len(run["Reco"]["SignalParticleSurface"]), dtype=bool
                    )
                )
                * (run["Reco"]["MatchedOpFlashPlane"] == 0)
                * (run["Reco"]["MatchedOpFlashPE"] > 0)
            )

            if this_fiducial_x == 0 and this_fiducial_y == 0 and this_fiducial_z == 0:
                mask = quality_mask
            else:
                mask = (
                    quality_mask
                    * (
                        np.absolute(run["Reco"]["RecoX"]) > this_fiducial_x
                        if config == "hd_1x2x6_lateralAPA"
                        else (
                            np.absolute(run["Reco"]["RecoX"])
                            < detector_x / 2 - this_fiducial_x
                            if config == "hd_1x2x6_centralAPA"
                            # VD workspace geometry simulates one drift only (upper X boundary);
                            # the real detector is symmetric about X = -DETECTOR_MAX_X.
                            # Fiducialize from the top boundary only.
                            else run["Reco"]["RecoX"] < detector_x / 2 - this_fiducial_x
                        )
                    )
                    * (
                        np.absolute(run["Reco"]["RecoY"])
                        < detector_y / 2 - this_fiducial_y
                    )
                    * (run["Reco"]["RecoZ"] > this_fiducial_z - info["DETECTOR_GAP_Z"])
                    * (
                        run["Reco"]["RecoZ"]
                        < info["DETECTOR_SIZE_Z"]
                        + info["DETECTOR_GAP_Z"]
                        - this_fiducial_z
                    )
                )

            idx_mask = np.where(mask == True)
            h, bins = np.histogram(
                run["Reco"][energy][idx_mask], bins=reco_energy_edges
            )

            # If there is some 0 entries between non-zero entries, dristribute counts from left to right to fill the gaps
            mc_counts = fill_gaps(h)
            h_rel_error_high = (
                np.sqrt(h) / h if "marley" in args.name else 0.02 + np.sqrt(h) / h
            )  # 2% systematic + Poisson + std uncertainty (+ std_uncertainty(h) / h) from adjacent bins
            h_rel_error_low = (
                np.sqrt(h) / h if "marley" in args.name else 0.02 + np.sqrt(h) / h
            )
            h, bins = np.histogram(
                run["Reco"][energy][idx_mask],
                bins=reco_energy_edges,
                weights=run["Reco"][
                    f"{weight}OscMean" if "marley" in args.name else weight
                ][idx_mask],
            )

            h = fill_gaps(h)
            h_error_high = h * h_rel_error_high
            h_error_low = h * h_rel_error_low
            h_error_high[np.isnan(h_error_high)] = 0
            h_error_low[np.isnan(h_error_low)] = 0

            plot_list.append(
                {
                    "Name": name,
                    "Component": weight_labels,
                    "Type": "signal" if "marley" in name else "background",
                    "MCCounts": mc_counts.tolist(),
                    "Counts": h.tolist(),
                    "Energy": reco_energy_centers,
                    "Error+": h_error_high.tolist(),
                    "Error-": h_error_low.tolist(),
                    "Color": color,
                    "FiducializedX": int(this_fiducial_x),
                    "FiducializedY": int(this_fiducial_y),
                    "FiducializedZ": int(this_fiducial_z),
                }
            )

        save_df(
            pd.DataFrame(plot_list),
            f"{data_path}/{args.folder.lower()}",
            config=config,
            name=name,
            filename=f"{energy}_Fiducial_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
