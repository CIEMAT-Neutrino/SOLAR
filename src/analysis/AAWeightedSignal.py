import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/nhits"
data_path = f"{root}/data/solar/nhits"

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

parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}

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
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
    },
    "yzoom": {"marley": [0, 6], "neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8]},
    "rewrite": True,
    "debug": True,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
    debug=user_input["debug"],
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
        }
        if "marley" in args.name
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    rm_branches=False,
    workflow=user_input["workflow"],
    debug=user_input["debug"],
)

nhits_list = []
for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    for name in configs[config]:
        for (weight, weight_labels, color), surface in track(
            product(
                zip(
                    user_input["weights"][name.split("_")[0]],
                    user_input["weight_labels"][name.split("_")[0]],
                    user_input["colors"][name.split("_")[0]],
                ),
                [None, -1, 0, 1, 2, 3, 4],
            ),
            total=(3 if "marley" in args.name else 1),
            description=f"Processing {name} - {config}",
        ):
            if surface is None:
                if args.folder.lower() in ["reduced", "truncated"]:
                    mask = run["Reco"]["SignalParticleSurface"] < 3
                else:
                    mask = np.ones(len(run["Reco"]["Event"]), dtype=bool)

            else:
                mask = run["Reco"]["SignalParticleSurface"] == surface

            nhits_list.append(
                {
                    "Config": config,
                    "Name": name,
                    "Folder": args.folder,
                    "Component": weight_labels,
                    "Weight": weight,
                    "Type": "signal" if "marley" in name else "background",
                    "Surface": surface,
                    "#Hits": run["Reco"]["NHits"][mask],
                    "Counts": run["Reco"][weight][mask],
                    "TrueEnergy": run["Reco"]["SignalParticleK"][mask],
                    "RecoEnergy": run["Reco"]["SolarEnergy"][mask],
                }
            )

save_df(
    pd.DataFrame(nhits_list),
    f"{data_path}/{args.folder.lower()}",
    config=config,
    name=name,
    filename=f"Weighted_{args.folder}",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)
