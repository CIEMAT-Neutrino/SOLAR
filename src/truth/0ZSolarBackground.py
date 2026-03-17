import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

figure_path = f"{root}/images/background"
data_path = f"{root}/data/background"

for save_path in [figure_path, data_path]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of background particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--names",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["all", "gamma", "neutron"]
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
user_input = {
    "workflow": "BACKGROUND",
    "rewrite": args.rewrite,
    "debug": args.debug,
}

runs = {}
data_list = []
pdg_list, id_list = [], []
detector_exposures = {}
pdg_label_dict = {"e-": "electron", "gamma": "gamma", "n": "neutron", "He4": "alpha"}

info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]
lar_density = 1.396  # g/cm^3

for name in args.names:
    configs = {args.config: [name]}
    run, output = load_multi(
        configs,
        load_all=False,
        preset=user_input["workflow"],
        name_prefix="MCParticle_",
        debug=user_input["debug"],
    )
    
    runs[name] = run
    
    files = len(run["Config"]["Geometry"][np.where(run["Config"]["Name"] == name)])
    events = len(run["Truth"]["Event"][np.where(run["Truth"]["Name"] == name)])
    particles = len(run["Reco"]["Event"][np.where(run["Reco"]["Name"] == name)])
    print(f"Loaded {files} files, {events} events, and {particles} particles")
    
    detector_time = 2 * info["TIMEWINDOW"] * events / 60 / 60 / 24 / 365  # years
    detector_mass = detector_x * detector_y * detector_z * lar_density / 1e9  # kT
    detector_exposure = detector_mass * detector_time  # kT*years
    detector_exposures[(args.config, name)] = detector_exposure
    rprint(
        f"{args.config} {name} exposure: {particles/detector_exposure:.2e} Counts (kT·years)⁻¹"
    )
    # pdg_list = np.unique(run["Reco"]["ParticlePDG"])
    # id_list = np.unique(run["Reco"]["ParticleLabelID"])
    pdg_list += list(np.unique(run["Reco"]["ParticlePDG"]))
    id_list += list(np.unique(run["Reco"]["ParticleLabelID"]))

# Remove duplicates from pdg_list and id_list
pdg_list = list(set(pdg_list))
id_list = list(set(id_list))

info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
gen_dict = get_gen_label(configs, debug=user_input["debug"])

gen_labels = [
    gen_dict[(info["GEOMETRY"], args.config, particle_id + 1)] for particle_id in id_list
]

inv_gen_dict = {v: k for k, v in gen_dict.items()}
colors = list(get_gen_color(gen_labels, debug=user_input["debug"]).values())
simple_id_dict = get_simple_names(gen_labels, debug=user_input["debug"])
# Create a dict where the keys are the unique arguments of the simple_id_dict and the values are lists of the corresponding keys
simple_id_dict_inv = {}
for key, value in simple_id_dict.items():
    simple_id_dict_inv.setdefault(value, []).append(key)

simple_id_dict_ids = {}
for key, value in simple_id_dict_inv.items():
    simple_id_dict_ids[key] = [inv_gen_dict[i][2] for i in value]

simple_id_dict_colors = {}
for key, value in simple_id_dict_inv.items():
    simple_id_dict_colors[key] = list(
        get_gen_color(value, debug=user_input["debug"]).values()
    )

pdg_color = get_pdg_color(pdg_list)
eff_flux = get_detected_solar_spectrum(
    bins=energy_centers, components=["b8", "hep"]
)
b8_eff_flux = get_detected_solar_spectrum(bins=energy_centers, components=["b8"])
hep_eff_flux = get_detected_solar_spectrum(bins=energy_centers, components=["hep"])

for idx, name in enumerate(args.names):
    # filter = run["Reco"]["Name"] == name
    for i, (
        (particle_simple_label, particle_simple_ids),
        (_, particle_simple_colors),
    ) in enumerate(zip(simple_id_dict_ids.items(), simple_id_dict_colors.items())):
        
        this_filter = np.zeros(len(runs[name]["Reco"]["ParticleLabelID"]), dtype=bool)
        for j, particle_id in enumerate(particle_simple_ids):
            this_filter += runs[name]["Reco"]["ParticleLabelID"] == particle_id - 1

        if np.sum(this_filter) < 1000:
            continue

        h, bins = np.histogram(
            runs[name]["Reco"]["ParticleK"][this_filter], bins=energy_edges[:81]
        )
        
        h = h / detector_exposures[(args.config, name)]
        h = h / ebin

        inserted = False
        for particle_simple_id in particle_simple_ids:
            for data in data_list:
                if (
                    particle_simple_id in data["Name"]
                    and data["Plot"] == "ParticleID"
                ):
                    inserted = True
                    break

        if inserted:
            print(f"Combining {particle_simple_label} in the plot")
            # Take the average of the counts where both are non-zero
            i = np.where(h != 0)
            j = np.where(data["Counts"] != 0)
            k = np.intersect1d(i, j)
            data["Counts"][k] = (data["Counts"][k] + h[k]) / 2
            # Simply insert the new data where the old data is zero
            l = np.where(data["Counts"] == 0)
            data["Counts"][l] = h[l]
        else:
            print(f"Adding {particle_simple_label} to the plot")
            data_list.append(
                {
                    "Config": args.config,
                    "Counts": h,
                    "Energy": energy_centers[:80],
                    "PDG": None,
                    "Particle": particle_simple_label,
                    "Name": particle_simple_ids,
                    "Color": particle_simple_colors[0],
                    "Plot": "ParticleID",
                    "LegendGroup": "DUNE Bkg.v3",
                }
            )

    for pdg in pdg_list:
        this_filter = runs[name]["Reco"]["ParticlePDG"] == pdg
        if np.sum(this_filter) < 1000:
            continue
        color = pdg_color[str(pdg)]
        print(pdg, color)
        pdg_label = Particle.from_pdgid(pdg).name
        h, bins = np.histogram(
            runs[name]["Reco"]["ParticleK"][this_filter], bins=energy_edges[:81]
        )
        h = h / detector_exposures[(args.config, name)]
        h = h / ebin
        # Check if the particle has already added to the plot
        inserted = False
        for data in data_list:
            if (
                f"{pdg_label_dict[pdg_label]}" == data["PDG"]
                and data["Plot"] == "ParticlePDG"
            ):
                inserted = True
                break

        if inserted:
            print(f"Combining {pdg_label} in the plot")
            # Take the average of the counts where both are non-zero
            i = np.where(h != 0)
            j = np.where(data["Counts"] != 0)
            k = np.intersect1d(i, j)
            data["Counts"][k] = (data["Counts"][k] + h[k]) / 2
            l = np.where(data["Counts"] == 0)
            data["Counts"][l] = h[l]
        else:
            print(f"Adding {pdg_label} to the plot")
            data_list.append(
                {
                    "Config": args.config,
                    "Counts": h,
                    "Energy": energy_centers[:80],
                    "PDG": f"{pdg_label_dict[pdg_label]}",
                    "Particle": None,
                    "Name": particle_simple_ids,
                    "Color": color,
                    "Plot": "ParticlePDG",
                    "LegendGroup": "DUNE Bkg.v3",
                }
            )

for jdx, (solar_label, solar_flux, solar_color) in enumerate(
    zip(
        [None, "8B", "HEP"],
        [eff_flux, b8_eff_flux, hep_eff_flux],
        ["rgb(102,102,102)", "rgb(225,124,5)", "rgb(204,80,62)"],
    )
):
    data_list.append(
        {
            "Config": args.config,
            "Counts": solar_flux[:80] * 60 * 60 * 24 * 365,
            "Energy": energy_centers[:80],
            "PDG": "neutrino" if jdx == 0 else None,
            "Particle": solar_label,
            "Name": None,
            "Color": solar_color,
            "Plot": "ParticlePDG" if jdx == 0 else "ParticleID",
            "LegendGroup": "CC B16-GS98",
        }
    )

save_df(
    pd.DataFrame(data_list),
    data_path,
    args.config,
    None,
    None,
    filename="Background",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)