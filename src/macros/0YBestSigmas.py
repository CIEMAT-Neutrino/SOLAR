import sys

sys.path.insert(0, "../../")

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
    "--folder", type=str, help="The name of the results folder", default="Reduced"
)
parser.add_argument(
    "--fiducial",
    nargs="+",
    type=int,
    help="The min niht cut for the analysis",
    default=[0, 20, 40, 60, 80, 100, 120],
)
parser.add_argument(
    "--nhits",
    nargs="+",
    type=int,
    help="The min niht cut for the analysis",
    default=nhits[:10],
)
parser.add_argument(
    "--ophits",
    nargs="+",
    type=int,
    help="The min ophit cut for the analysis",
    default=nhits[3:10],
)
parser.add_argument(
    "--adjcls",
    nargs="+",
    type=int,
    help="The max adjcl cut for the analysis",
    default=nhits[::-1][10:],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

analysis = parser.parse_args().analysis
reference = parser.parse_args().reference
config = parser.parse_args().config
name = parser.parse_args().name
folder = parser.parse_args().folder

fiducials = parser.parse_args().fiducial
hits = parser.parse_args().nhits
ophits = parser.parse_args().ophits
adjcls = parser.parse_args().adjcls

configs = {config: [name]}

user_input = {
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}

# results = dict()
fastest_sigma2, fastest_sigma3, highest_sigma = {}, {}, {}
fastest_sigmas = [fastest_sigma2, fastest_sigma3]

# thld_idx = np.where(rebin_centers > user_input["threshold"])[0][0]
plot_data = {}
for config in configs:
    for idx, name in enumerate(configs[config]):
        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}/{config}_{name}_{analysis}_Results.pkl",
        )
        sigmas_df = explode(
            sigmas_df, ["Sigma2", "Sigma3", "Exposure", "Gaussian", "Asimov"]
        )

        for energy_label in ["Cluster", "Total", "Selected", "Solar"]:
            rprint(f"Evaluating {energy_label}Energy")
            for idx, sigma_label in enumerate(["Sigma2", "Sigma3"]):
                # Find the entry with the highest significance (max this_sigma_df["sigma_label"])
                this_sigma_df = sigmas_df[
                    (sigmas_df["Config"] == config)
                    * (sigmas_df["Name"] == name)
                    * (sigmas_df["EnergyLabel"] == energy_label)
                    * (sigmas_df["Fiducialized"].isin(fiducials))
                    * (sigmas_df["NHits"].isin(nhits))
                    * (sigmas_df["OpHits"].isin(ophits))
                    * (sigmas_df["AdjCl"].isin(adjcls))
                ]
                # print(this_sigma_df)
                if idx == 0:
                    this_sigma = this_sigma_df.loc[
                        this_sigma_df[reference] == this_sigma_df[reference].max()
                    ].copy()

                    if len(this_sigma) >= 1:
                        this_sigma = this_sigma.iloc[0]
                    elif len(this_sigma) == 0:
                        pass

                    if isinstance(this_sigma["Fiducialized"], np.int64):
                        highest_sigma[(config, name, energy_label)] = {
                            sigma_label: this_sigma[sigma_label],
                            "Values": this_sigma[reference],
                            "Fiducialized": this_sigma["Fiducialized"],
                            "NHits": this_sigma["NHits"],
                            "OpHits": this_sigma["OpHits"],
                            "AdjCl": this_sigma["AdjCl"],
                        }
                        rprint(
                            f'\t*Adding highest sigma with fiducial {this_sigma["Fiducialized"]} nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
                        )
                    else:
                        rprint(f"\t[yellow][WARNING] Not found highest sigma[/yellow]")

                # Find the entry with the fastest sigma (min this_sigma_df["sigma_label"])
                this_sigma = this_sigma_df.loc[
                    (this_sigma_df[sigma_label] > 0)
                    * (this_sigma_df[reference] == this_sigma_df[reference].max())
                ].copy()

                if len(this_sigma) >= 1:
                    this_sigma = this_sigma.iloc[0]
                elif len(this_sigma) == 0:
                    pass

                if isinstance(this_sigma["Fiducialized"], np.int64):
                    fastest_sigmas[idx][(config, name, energy_label)] = {
                        sigma_label: this_sigma[sigma_label],
                        "Values": this_sigma[reference],
                        "Fiducialized": this_sigma["Fiducialized"],
                        "NHits": this_sigma["NHits"],
                        "OpHits": this_sigma["OpHits"],
                        "AdjCl": this_sigma["AdjCl"],
                    }
                    rprint(
                        f'\t*Adding fastest sigma{idx+2} with fiducial {this_sigma["Fiducialized"]} nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
                    )
                else:
                    rprint(
                        f"\t[yellow][WARNING] Not found fastest sigma{idx+2}[/yellow]"
                    )

        for sigma_results, sigma_results_label in zip(
            [highest_sigma, fastest_sigma2, fastest_sigma3],
            ["highest", "fastest_sigma2", "fastest_sigma3"],
        ):
            save_df(
                pd.DataFrame(sigma_results),
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}",
                config,
                name,
                filename=f"{sigma_results_label}_{analysis}",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
