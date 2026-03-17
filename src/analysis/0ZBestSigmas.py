import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/results"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--analysis",
    type=str,
    help="The name of the analysis",
    choices=["DayNight", "Sensitivity", "HEP"],
    required=True,
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference column",
    choices=["Asimov", "Gaussian"],
    default="Asimov",
    required=True,
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configuration to load",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["marley"],
)
parser.add_argument(
    "--folder", type=str, help="The name of the results folder", default="Nominal"
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
parser.add_argument(
    "--nhits",
    type=int,
    help="The min niht cut for the analysis",
    default=None,
)
parser.add_argument(
    "--ophits",
    type=int,
    help="The min ophit cut for the analysis",
    default=None,
)
parser.add_argument(
    "--adjcls",
    type=int,
    help="The max adjcl cut for the analysis",
    default=None,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

fastest_sigma2, fastest_sigma3, highest_sigma = {}, {}, {}
fastest_sigmas = [fastest_sigma2, fastest_sigma3]

plot_data = {}
for config, name, energy_label in product(args.config, args.name, args.energy):
    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy_label}_{args.analysis}_Results.pkl",
    )
    sigmas_df = explode(sigmas_df, ["Sigma2", "Sigma3", "Exposure", args.reference])

    rprint(f"Evaluating {energy_label}")
    for idx, sigma_label in enumerate(["Sigma2", "Sigma3"]):
        # Find the entry with the highest significance (max this_sigma_df["sigma_label"])
        this_sigma_df = sigmas_df[
            (sigmas_df["Config"] == config)
            * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"].isin(nhits[:10]))
            * (sigmas_df["OpHits"].isin(nhits[3:10]))
            * (sigmas_df["AdjCl"].isin(nhits[::-1][10:]))
        ]
        # print(this_sigma_df)
        if idx == 0:
            this_sigma = this_sigma_df.loc[
                this_sigma_df[args.reference] == this_sigma_df[args.reference].max()
            ].copy()

            if len(this_sigma) >= 1:
                this_sigma = this_sigma.iloc[0]
            elif len(this_sigma) == 0:
                pass

            highest_sigma[(config, name, energy_label)] = {
                sigma_label: this_sigma[sigma_label],
                "Values": this_sigma[args.reference],
                "NHits": this_sigma["NHits"],
                "OpHits": this_sigma["OpHits"],
                "AdjCl": this_sigma["AdjCl"],
            }
            rprint(
                f'\t*Adding highest sigma with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
            )

        # Find the entry with the fastest sigma (min this_sigma_df["sigma_label"])
        this_sigma = this_sigma_df.loc[
            (this_sigma_df[sigma_label] > 0)
            * (this_sigma_df[args.reference] == this_sigma_df[args.reference].max())
        ].copy()

        if len(this_sigma) >= 1:
            this_sigma = this_sigma.iloc[0]
        elif len(this_sigma) == 0:
            pass

        fastest_sigmas[idx][(config, name, energy_label)] = {
            sigma_label: this_sigma[sigma_label],
            "Values": this_sigma[args.reference],
            "NHits": this_sigma["NHits"],
            "OpHits": this_sigma["OpHits"],
            "AdjCl": this_sigma["AdjCl"],
        }
        rprint(
            f'\t*Adding fastest sigma{idx+2} with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
        )

    for sigma_results, sigma_results_label in zip(
        [highest_sigma, fastest_sigma2, fastest_sigma3],
        ["highest", "fastest_sigma2", "fastest_sigma3"],
    ):
        # If file already exists, load it and update it with the new results, otherwise create a new file

        save_df(
            pd.DataFrame(sigma_results),
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}",
            config,
            name,
            filename=f"{sigma_results_label}_{args.analysis}",
            rm=args.rewrite,
            debug=args.debug,
        )
