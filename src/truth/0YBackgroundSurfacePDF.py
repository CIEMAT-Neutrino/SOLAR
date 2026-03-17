import os
import sys
import time

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


def compute_weights(p_sim, surface_id_sim, hist, bins, alpha_truth):
    n_s = 0
    w = np.zeros_like(p_sim)
    for s in np.unique(surface_id_sim):
        mask = surface_id_sim == s
        if s < 0 or len(hist[s]) == 0:
            continue

        n_s += 1
        p_s = p_sim[mask]
        bin_idx_s = np.digitize(p_s, bins[s]) - 1
        bin_idx_s = np.clip(bin_idx_s, 0, len(hist[s]) - 1)
        w[mask] = alpha_truth[s] * hist[s][bin_idx_s]

    return n_s * w


figure_path = f"{root}/images/background"
data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/legacy"
for save_path in [figure_path, data_path]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Save PDFs of the background momentum distributions for the solar analysis"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name",
    type=str,
    help="The name of the configuration",
    default="gamma",
    choices=["gamma", "neutron"],
)
parser.add_argument(
    "--binwidth",
    type=float,
    help="The bin width for the histograms in (MeV)",
    default=None,
)
parser.add_argument(
    "--pdf_floor",
    type=float,
    help="The minimum value for the PDF to avoid zeros",
    default=1e-6,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

configs = {args.config: [args.name]}
user_input = {
    "workflow": "BACKGROUND",
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, load_all=True, name_prefix="MCParticle_", debug=user_input["debug"]
)

run, output, new_branches = compute_particle_surface(
    run, configs, {}, ["Reco"], "Particle", False, output, user_input["debug"]
)
rprint(output)

A = []
hist_s = []
bins_s = []
alpha_truth = []
exposure_dict = {}
files = len(run["Config"]["Geometry"])
events = len(run["Truth"]["Event"])
particles = len(run["Reco"]["Event"])
print(f"Loaded {files} files, {events} events, and {particles} particles")

surfaces = json.load(open(f"{root}/import/surface_positions.json"))

info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
detector_time = 2 * info["TIMEWINDOW"] * events / 60 / 60 / 24 / 365.25  # years
detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]
lar_density = 1.396  # g/cm^3
detector_mass = detector_x * detector_y * detector_z * lar_density / 1e9  # kT
detector_exposure = detector_mass * detector_time  # kT*years
print(f"Detector exposure: {detector_exposure} kT·years")
data = dict()


for variable, (surface_label, [surface_value, surface_id]) in product(
    ["ParticleP", "ParticleX", "ParticleY", "ParticleZ"],
    surfaces[info["GEOMETRY"]].items(),
):
    if int(surface_id) < 0:
        continue

    filtered_run, mask, output = compute_filtered_run(
        run,
        configs,
        params={
            ("Reco", "SignalParticleSurface"): ("equal", surface_id),
        },
        output=output,
        debug=user_input["debug"],
    )
    if len(filtered_run["Reco"]["ParticleP"]) > 0:
        if variable == "ParticleP":
            # Find best bin width using Freedman-Diaconis rule
            if args.binwidth is None:
                mask = ~np.isnan(filtered_run["Reco"][variable])
                q75, q25 = np.percentile(filtered_run["Reco"][variable][mask], [75, 25])
                iqr = q75 - q25
                binwidth = 2 * iqr / np.cbrt(len(filtered_run["Reco"][variable][mask]))
            else:
                binwidth = args.binwidth
            bins = np.arange(
                np.min(filtered_run["Reco"][variable]),
                np.max(filtered_run["Reco"][variable]) + binwidth,
                binwidth,
            )

        else:
            bins = np.linspace(
                np.min(filtered_run["Reco"][variable]),
                np.max(filtered_run["Reco"][variable]),
                100,
            )

        h, bins = np.histogram(
            filtered_run["Reco"][variable],
            bins=bins,
            density=True,
        )

        data[(surface_id, variable)] = (h, bins)

    else:
        rprint(
            f"[yellow][WARNING][/yellow] No particles found for surface {surface_label} and variable {variable}. Skipping PDF computation for this surface."
        )
        h = np.array([])
        bins = np.array([])
        data[(surface_id, variable)] = (h, bins)

    if (
        (int(surface_id) == 0 and variable == "ParticleP" and len(h) > 0)
        or (int(surface_id) == 0 and variable == "ParticleX" and len(h) > 0)
        or (int(surface_id) in [1, 2] and variable == "ParticleY" and len(h) > 0)
        or (int(surface_id) in [3, 4] and variable == "ParticleZ" and len(h) > 0)
    ):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=(bins[:-1] + bins[1:]) / 2,
                y=h / detector_exposure,
                mode="lines",
                name=args.name,
                line_shape="hvh",
            )
        )
        # Add vline for the median of the distribution
        median_value = np.median(filtered_run["Reco"][variable])
        if np.isnan(median_value):
            median_value = np.mean(filtered_run["Reco"][variable])

        fig.add_vline(
            x=median_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Median = {:.2f}".format(median_value),
            annotation_position="top right",
        )
        fig.update_yaxes(title_text=f"Events / (kT·years)")
        fig.update_xaxes(title_text=f"{variable}")
        fig = format_coustom_plotly(
            fig,
            log=(False, variable == "ParticleP"),
            title=f"Surface {surface_label} ({surface_label})",
            tickformat=(".1f", ".0e"),
            add_watermark=False,
        )
        save_figure(
            fig,
            figure_path,
            args.config,
            args.name,
            filename=f"{surface_label.replace(' ', '_')}_{variable.lower()}_PDF",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    # Save exposure information to pkl
    if variable == "ParticleP":
        counts = len(filtered_run["Reco"]["ParticleP"])
        this_exposure_dict = {
            "detector_mass": detector_mass,
            "detector_time": detector_time,
            "exposure": detector_exposure,
            "surface_label": surface_label,
            "surface": surface_id,
            "counts": counts,
            "name": args.name,
        }
        exposure_dict[int(surface_id)] = this_exposure_dict
        rprint(
            f"Surface {surface_label} - Counts: {counts}, Exposure: {detector_exposure:.2f} kT·years, Rate: {counts/detector_exposure:.2e} events/(kT·years)"
        )

for surface_label, (surface_value, surface_id) in surfaces[info["GEOMETRY"]].items():
    if int(surface_id) < 0:
        continue

    eps = args.pdf_floor
    hist = np.maximum(data[(int(surface_id), "ParticleP")][0], eps)
    bins = data[(int(surface_id), "ParticleP")][1]
    # print(
    #     f"Surface {surface_label} - PDF sum check: {np.sum(hist * np.diff(bins)):.2e}"
    # )
    hist /= np.sum(hist * np.diff(bins))

    hist_s.append(hist)
    bins_s.append(bins)

    if int(surface_id) < 0:
        continue
    if int(surface_id) == 0:
        A.append((info["PRODUCTION_SIZE_Y"]) * (info["PRODUCTION_SIZE_Z"]))
    elif int(surface_id) in [1, 2]:
        A.append((info["PRODUCTION_SIZE_X"]) * (info["PRODUCTION_SIZE_Z"]))
    elif int(surface_id) in [3, 4]:
        A.append((info["PRODUCTION_SIZE_X"]) * (info["PRODUCTION_SIZE_Y"]))
    else:
        raise ValueError(f"Invalid surface_id: {surface_id}")
print(f"Surface areas: {A}")

save_pkl(
    exposure_dict,
    data_path,
    args.config,
    None,
    filename=f"{args.name}_exposure",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

save_pkl(
    (hist_s, bins_s),
    data_path,
    args.config,
    None,
    filename=f"{args.name}_pdf",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

# Define alpha_truth wich is the fraction of particles per surface. This can be computed from the counts in the exposure_dict for each surface and the total counts
total_counts = sum([exposure["counts"] for exposure in exposure_dict.values()])

for exposure in exposure_dict.values():
    alpha_truth.append(exposure["counts"] / total_counts)
print(f"Alpha truth: {alpha_truth}")

sample = 1000
scale = 1000

# Produce a random sample of (p, u, v) for all surfaces
s_test, u_test, v_test = np.array([]), np.array([]), np.array([])
for s in range(np.max(list(exposure_dict.keys())) + 1):
    if s == 0:
        s_s = np.zeros(sample)
        u_s = np.random.uniform(
            -info["PRODUCTION_SIZE_Y"], info["PRODUCTION_SIZE_Y"], sample
        )
        v_s = np.random.uniform(0, info["PRODUCTION_SIZE_Z"], sample)
    elif s in [1, 2]:
        s_s = np.ones(sample) * s
        u_s = np.random.uniform(
            -info["PRODUCTION_SIZE_X"], info["PRODUCTION_SIZE_X"], sample
        )
        v_s = np.random.uniform(0, info["PRODUCTION_SIZE_Z"], sample)
    elif s in [3, 4]:
        s_s = np.ones(sample) * s
        u_s = np.random.uniform(
            -info["PRODUCTION_SIZE_X"] / 2, info["PRODUCTION_SIZE_X"] / 2, sample
        )
        v_s = np.random.uniform(
            -info["PRODUCTION_SIZE_Y"] / 2, info["PRODUCTION_SIZE_Y"] / 2, sample
        )
    else:
        raise ValueError(f"Invalid surface_id: {s}")
    if s == 0:
        s_test = s_s
        u_test = u_s
        v_test = v_s
    else:
        s_test = np.concatenate([s_test, s_s])
        u_test = np.concatenate([u_test, u_s])
        v_test = np.concatenate([v_test, v_s])

p_test = np.random.uniform(0, 14, sample * 5)
start = time.time()
w = compute_weights(p_test, s_test.astype(int), hist_s, bins_s, alpha_truth)
end = time.time()

rprint(f"PDF binning took {scale*(end - start)/60:.2f} min for {sample*scale} samples")

for surface_label, (surface_value, surface_id) in surfaces[info["GEOMETRY"]].items():
    if surface_id < 0:
        continue
    w_surface = w[s_test.astype(int) == surface_id]
    p_surface = p_test[s_test.astype(int) == surface_id]
    h_weighted, bins_weighted = np.histogram(
        p_surface, bins=bins_s[surface_id], weights=w_surface
    )
    bin_centers = (bins_s[surface_id][:-1] + bins_s[surface_id][1:]) / 2

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=h_weighted,
            mode="lines",
            name="Weighted by PDF",
            line_shape="hvh",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=data[(surface_id, "ParticleP")][0] / detector_exposure,
            mode="lines",
            name="Original PDF",
            line_shape="hvh",
        )
    )
    fig.update_yaxes(title_text=f"Events / (kT·years)")
    fig.update_xaxes(title_text=f"ParticleP (MeV)")
    fig = format_coustom_plotly(
        fig,
        title=f"Surface {surface_label} ({surface_label}) - PDF Check",
        tickformat=(".1f", ".0e"),
        log=(False, True),
        ranges=(None, (-3, None)),
    )
    save_figure(
        fig,
        figure_path,
        args.config,
        args.name,
        filename=f"{surface_label.replace(' ', '_')}_pdf_check",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
