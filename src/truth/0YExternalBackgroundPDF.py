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
data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth"
for save_path in [figure_path, data_path]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Save PDFs of the background momentum distributions for the solar analysis"
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configuration to load",
    default=[
        "hd_1x2x6_centralAPA",
        "hd_1x2x6_lateralAPA",
        "vd_1x8x14_3view_30deg_nominal",
        "vd_1x8x14_3view_30deg_shielded",
    ],
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["gamma"],
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
    default=1e-8,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

for config, name in product(args.config, args.name):
    data = dict()
    A = []
    hist_s = []
    bins_s = []
    alpha_truth = []
    exposure_dict = {}
    bin_edges = np.array([])
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]
    lar_density = 1.396  # g/cm^3
    detector_mass = detector_x * detector_y * detector_z * lar_density / 1e9  # kT

    areas = json.load(open(f"{root}/import/surface_areas.json", "r"))
    for geometry, surfaces in areas.items():
        for surface, area in surfaces.items():
            areas[geometry][surface] = eval(area)

    files = json.load(open(f"{root}/import/surface_activity.json", "r"))[config]
    surface_names = json.load(open(f"{root}/import/surfaces.json", "r"))
    surfaces = json.load(open(f"{root}/import/surface_positions.json"))
    rprint(f"Loaded config {config} with surfaces: {files.keys()}")

    integrated_flux = {}
    combined_spectra = {}
    for surface in files.items():
        integrated_flux[surface[0]] = 0
        combined_spectra[surface[0]] = []
        for particle_origin, (this_file, weight) in surface[1].items():
            particle_type = particle_origin.lower().split("_")[-1]
            if name not in particle_type.lower():
                continue

            file_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{this_file}.root"
            # Open the root file
            root_file = TFile.Open(file_path)
            graph = root_file.Get("Gammas" if particle_type == "gamma" else "Neutrons")
            # Extract the original labels from the graph
            xlabel = graph.GetXaxis().GetTitle()
            ylabel = graph.GetYaxis().GetTitle()

            # Extract the data points from the graph
            k = []
            y = []
            for i in range(graph.GetN()):
                k.append(graph.GetX()[i])
                y.append(graph.GetY()[i])

            y = np.array(y) * weight / np.sum(y)  # Normalize the spectrum to the weight
            y = y * 60 * 60 * 24 * 365.25  # Convert from per second to per year
            y /= detector_mass  # Convert from per year to per kT·year
            if (
                config == "vd_1x8x14_3view_30deg_shielded"
                and particle_origin == "cavernwall_gamma"
            ):
                reduction = 0.284571618037  # Bq/cm^2
                rprint(
                    f"Applying reduction factor of {reduction:.3f} to {particle_origin} spectrum for surface {surface[0]} in config {config}"
                )
                # Apply the reduction factor to the spectrum above the 3.3 MeV threshold
                y = np.where(np.asarray(k) >= 3.3e3, y * reduction, y)

            area_weight = areas[config.split("_")[0]][surface[0]]
            integrated_flux[surface[0]] += float(
                np.trapezoid(y, k) * area_weight  # type: ignore
            )  # pyright: ignore[reportAttributeAccessIssue]
            # Convert kinetic energy array into momentum array assuming mass of neutron or gamma
            if particle_type == "electron":
                p = np.sqrt(
                    2 * 0.511e-3 * np.array(k)
                )  # Mass of electron is 0.511e3 MeV/c^2
            elif particle_type == "gamma":
                p = 1e-3 * np.array(k)  # For photons, momentum is equal to energy

            elif particle_type == "neutron":
                p = np.sqrt(
                    2 * 939.565e-3 * np.array(k)
                )  # Mass of neutron is 939.565e3 MeV/c^2
            else:
                raise ValueError(f"Invalid particle type: {particle_type}")

            # Combine the spectra by surface area
            if len(combined_spectra[surface[0]]) == 0:
                combined_spectra[surface[0]] = (
                    np.array(y) * area_weight,
                    p,
                )
            else:
                combined_spectra[surface[0]] = (
                    combined_spectra[surface[0]][0] + np.array(y) * area_weight,
                    combined_spectra[surface[0]][1],
                )
        if "gamma" in name:
            spectrum_data, momentum_data = combined_spectra[surface[0]]
            mask = momentum_data >= 4
            combined_spectra[surface[0]] = (spectrum_data[mask], momentum_data[mask])

    # rprint(f"Integrated fluxes for config {config} and name {name} - {integrated_flux}")

    for variable, (surface_label, [surface_value, surface_id]) in product(
        ["ParticleP"],
        surfaces[info["GEOMETRY"]].items(),
    ):
        if int(surface_id) < 0:
            continue
        surface_id = str(surface_id)

        if len(combined_spectra[surface_id][0]) > 0:
            if args.binwidth is None:
                binwidth = 0.02 * np.sqrt(np.mean(combined_spectra[surface_id][1] ** 2))
            else:
                binwidth = args.binwidth

            bin_edges = np.arange(
                np.min(combined_spectra[surface_id][1]) - binwidth,
                np.max(combined_spectra[surface_id][1]) + binwidth,
                binwidth,
            )
            h, bins = np.histogram(
                combined_spectra[surface_id][1],
                bins=bin_edges,
                weights=combined_spectra[surface_id][0],
            )
            data[(surface_id, "ParticleP")] = (h, bins)

        else:
            rprint(
                f"[yellow][WARNING][/yellow] No spectra found for surface {surface_label}. Skipping PDF computation for this surface."
            )
            h = np.array([])
            bins = np.array([])
            bin_edges = np.array([])
            data[(surface_id, "ParticleP")] = (h, bins)

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=(bins[:-1] + bins[1:]) / 2,
                y=h,
                mode="lines",
                name=name,
                line_shape="hvh",
            )
        )

        fig.update_yaxes(title_text=f"Events / (kT·years)")
        fig.update_xaxes(title_text=f"Particle Momentum (MeV)")
        fig = format_coustom_plotly(
            fig,
            log=(False, True),
            title=f"Surface {surface_label} ({surface_label})",
            tickformat=(".1f", ".0e"),
            add_watermark=False,
        )
        save_figure(
            fig,
            figure_path,
            config,
            name,
            subfolder="weights",
            filename=f"{surface_label.replace(' ', '_')}_particlep_PDF",
            rm=args.rewrite,
            debug=args.debug,
        )

        this_exposure_dict = {
            "detector_mass": detector_mass,
            "detector_time": 1,
            "exposure": 1,
            "surface_label": surface_label,
            "surface": surface_id,
            "counts": int(np.sum(h * np.diff(bins))),
            "name": args.name,
        }
        exposure_dict[int(surface_id)] = this_exposure_dict
        rprint(
            f"Surface {surface_label} - Rate: {this_exposure_dict['counts']/this_exposure_dict['exposure']:.2e} events/(kT·years)"
        )

    for surface_label, (surface_value, surface_id) in surfaces[
        info["GEOMETRY"]
    ].items():

        if int(surface_id) < 0:
            continue
        surface_id = str(surface_id)

        eps = args.pdf_floor
        hist = np.maximum(data[(str(surface_id), "ParticleP")][0], eps)
        bins = data[(str(surface_id), "ParticleP")][1]
        hist /= np.sum(hist * np.diff(bins))

        hist_s.append(hist)
        bins_s.append(bins)

    print(areas[config.split("_")[0]])

    save_pkl(
        exposure_dict,
        data_path,
        config,
        None,
        filename=f"{name}_exposure",
        rm=args.rewrite,
        debug=args.debug,
    )

    save_pkl(
        (hist_s, bins_s),
        data_path,
        config,
        None,
        filename=f"{name}_pdf",
        rm=args.rewrite,
        debug=args.debug,
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
                -info["PRODUCTION_SIZE_X"] / 2,
                info["PRODUCTION_SIZE_X"] / 2,
                sample,
            )
            v_s = np.random.uniform(
                -info["PRODUCTION_SIZE_Y"] / 2,
                info["PRODUCTION_SIZE_Y"] / 2,
                sample,
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

    k_test = np.random.uniform(0, 14, sample * 5)
    if "gamma" in name:
        p_test = k_test
    elif "neutron" in name:
        p_test = np.sqrt(2 * 939.565 * k_test)
    else:
        raise ValueError(f"Invalid name: {name}")

    start = time.time()
    w = compute_weights(p_test, s_test.astype(int), hist_s, bins_s, alpha_truth)
    end = time.time()

    rprint(
        f"PDF binning took {scale*(end - start)/60:.2f} min for {sample*scale} samples"
    )

    for surface_label, (surface_value, surface_id) in surfaces[
        info["GEOMETRY"]
    ].items():
        if surface_id < 0:
            continue
        w_surface = w[s_test.astype(int) == surface_id]
        p_surface = p_test[s_test.astype(int) == surface_id]

        h_weighted, bins_weighted = np.histogram(
            p_surface, bins=bin_edges, weights=w_surface
        )
        fig = make_subplots(rows=1, cols=1)

        y = (
            exposure_dict[surface_id]["counts"]
            * h_weighted
            / np.sum(h_weighted * np.diff(bins_weighted))
        )
        fig.add_trace(
            go.Scatter(
                x=data[(str(surface_id), "ParticleP")][1],
                y=y,
                mode="lines",
                name="Weighted by PDF",
                line_shape="hvh",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data[(str(surface_id), "ParticleP")][1],
                y=data[(str(surface_id), "ParticleP")][0],
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
        )
        save_figure(
            fig,
            figure_path,
            config,
            name,
            filename=f"{surface_label.replace(' ', '_')}_pdf_check",
            rm=args.rewrite,
            debug=args.debug,
        )
