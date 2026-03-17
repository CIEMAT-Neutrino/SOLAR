import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/truth"
data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal"

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
    "--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal"], default="Nominal",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

configs = {args.config: [args.name]}
user_input = {
    "rewrite": args.rewrite,
    "debug": args.debug,
}

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = 1e9  # kT
    eff_flux_b8 = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["b8"]
    )
    eff_flux_hep = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["hep"]
    )
    eff_flux_comb = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["b8", "hep"]
    )
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)

    (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
        dm2=None,
        sin13=None,
        sin12=None,
        path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
        ext="pkl",
        auto=True,
        debug=user_input["debug"],
    )

    for dm2, sin13, sin12 in track(zip(dm2_list, sin13_list, sin12_list), description="Processing oscillation parameters", total=len(dm2_list)):
        truth_oscillation_df = pd.read_pickle(
            f"{info['PATH']}/data/OSCILLATION/pkl/raw/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
        )
        for name, (eff_flux, label) in product(
            configs[config],
            zip((eff_flux_b8, eff_flux_hep, eff_flux_comb), ["b8", "hep", "comb"]),
        ):

            for azimuth in truth_oscillation_df.index[: len(truth_oscillation_df) // 2]:
                this_azimuth_flux = (
                    eff_flux
                    * 60
                    * 60
                    * 24
                    * 365
                    * np.array(list(truth_oscillation_df.loc[azimuth].values))
                )
                kde_azimuth = KernelDensity(
                    bandwidth=ebin, kernel="gaussian", algorithm="kd_tree"
                ).fit(energy_centers[:, None], sample_weight=this_azimuth_flux)
                save_pkl(
                    kde_azimuth,
                    f"{data_path}/osc/azimuth_{-azimuth:.3f}",
                    config,
                    None,
                    filename=f"{name}_{label}_kde_p_azimuth_{-azimuth:.3f}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )
                print(
                    f"KDE model saved for azimuth: {azimuth:.3f}, dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}"
                )