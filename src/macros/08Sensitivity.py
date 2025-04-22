import sys, json

sys.path.insert(0, "../../")
from lib import *

from lib.root_functions import Sensitivity_Fitter
from lib.osc_functions import get_oscillation_datafiles

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "HEP"],
    default="HEP",
    required=True,
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
    help="The name of the results folder",
    default="Reduced",
    choices=["Reduced", "Nominal"],
)
parser.add_argument(
    "--signal_uncertanty",
    type=float,
    help="The signal uncertanty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertanty",
    type=float,
    help="The background uncertanty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    default=["Cluster", "Total", "Selected", "Solar"],
)
parser.add_argument(
    "--fiducial", type=int, help="The fiducial cut for the analysis", default=None
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcl", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=10.0
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

config = args.config
name = args.name
configs = {config: [name]}

signal_uncertanty = args.signal_uncertanty
background_uncertanty = args.background_uncertanty

threshold = args.threshold
thld = np.where(sensitivity_rebin_centers > threshold)[0][0]


for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))

    fastest_sigma = pickle.load(
        open(
            f"{info['PATH']}/{args.reference.upper()}/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_fastest_sigma3_{args.reference}.pkl",
            "rb",
        )
    )

    for name, key in product(configs[config], fastest_sigma):
        if key[2] not in args.energy:
            continue
        else:
            energy = key[2]

        paths = {
            "signal_path": f"{info['PATH']}/SENSITIVITY/{config}/{name}/{args.folder.lower()}/{energy}Energy",
            "background_path": f"{info['PATH']}/SENSITIVITY/{config}/background/{args.folder.lower()}/{energy}Energy",
        }

        if args.fiducial is not None:
            fiducial = args.fiducial
        else:
            rprint(f"Using optimized fiducial cut {fastest_sigma[key]['Fiducialized']}")
            fiducial = int(fastest_sigma[key]["Fiducialized"])

        if args.nhits is not None:
            nhits = args.nhits
        else:
            rprint(f"Using optimized nhits {fastest_sigma[key]['NHits']}")
            nhits = int(fastest_sigma[key]["NHits"])

        if args.adjcl is not None:
            adjcl = args.adjcl
        else:
            rprint(f"Using optimized adjcl {fastest_sigma[key]['AdjCl']}")
            adjcl = int(fastest_sigma[key]["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            rprint(f"Using optimized ophits {fastest_sigma[key]['OpHits']}")
            ophits = int(fastest_sigma[key]["OpHits"])

        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            dm2=None,
            sin13=[0.021],
            sin12=None,
            path=f"{paths['signal_path']}/",
            ext="pkl",
            auto=False,
            debug=True,
        )
        react_sin13_df = pd.DataFrame(
            columns=np.unique(sin13_list), index=np.unique(dm2_list)
        )
        react_sin12_df = pd.DataFrame(
            columns=np.unique(sin12_list), index=np.unique(dm2_list)
        )
        solar_sin13_df = pd.DataFrame(
            columns=np.unique(sin13_list), index=np.unique(dm2_list)
        )
        solar_sin12_df = pd.DataFrame(
            columns=np.unique(sin12_list), index=np.unique(dm2_list)
        )

        react_df = pd.DataFrame(columns=["dm2", "sin13", "sin12", "chi2"])
        solar_df = pd.DataFrame(columns=["dm2", "sin13", "sin12", "chi2"])

        solar_tuple = (
            analysis_info["SOLAR_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
        react_tuple = (
            analysis_info["REACT_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
        pred1_df = pd.read_pickle(
            f'{paths["signal_path"]}/{config}_{name}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["SOLAR_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
        )
        # Fill NaN values with 0 using numpy function nan_to_num
        pred1_df = np.nan_to_num(pred1_df, nan=0.0)
        pred2_df = pd.read_pickle(
            f'{paths["signal_path"]}/{config}_{name}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["REACT_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
        )
        pred2_df = np.nan_to_num(pred2_df, nan=0.0)
        bkg_df = pd.read_pickle(
            f'{paths["background_path"]}/{config}_background_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl'
        )
        bkg_df = np.nan_to_num(bkg_df, nan=0.0)

        fake_df_dict = {}
        for dm2, sin13, sin12 in track(
            zip(dm2_list, sin13_list, sin12_list),
            description="Loading data...",
            total=len(dm2_list),
        ):
            this_df = (
                pd.read_pickle(
                    f'{paths["signal_path"]}/{config}_{name}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl'
                )
                + bkg_df
            )

            this_df = np.nan_to_num(this_df, nan=0.0)
            fake_df_dict[(dm2, sin13, sin12)] = this_df

        # Set initial parameter values for the fit
        initial_A_pred = 0.0
        initial_A_bkg = 0.0

        for i in track(
            range(len(fake_df_dict.keys())), description="Computing data..."
        ):
            params = list(fake_df_dict.keys())[i]
            obs_df = list(fake_df_dict.values())[i]

            rprint(
                "\n--------------------------------",
                "\n# Parameter Combination " + str(i) + ": ",
                params,
            )

            # Perform the fit
            fitter = Sensitivity_Fitter(
                obs_df[:, thld:],
                pred1_df[:, thld:],
                bkg_df[:, thld:],
                SigmaPred=signal_uncertanty,
                SigmaBkg=background_uncertanty,
            )

            chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
            rprint(f"Solar Chi2: {chi2:.2f}")

            if params[2] == analysis_info["SIN12"]:
                rprint("Saving data to sin13")
                solar_sin13_df.loc[params[0], params[1]] = chi2

            if params[1] == analysis_info["SIN13"]:
                rprint("Saving data to sin12")
                solar_sin12_df.loc[params[0], params[2]] = chi2

            if (
                params[2] != analysis_info["SIN12"]
                and params[1] != analysis_info["SIN13"]
            ):
                rprint(f"[cyan]INFO: sin12 and sin13 do not match default[/cyan]")

            solar_df.loc[i] = [params[0], params[1], params[2], chi2]

            fitter = Sensitivity_Fitter(
                obs_df[:, thld:],
                pred2_df[:, thld:],
                bkg_df[:, thld:],
                SigmaPred=signal_uncertanty,
                SigmaBkg=background_uncertanty,
            )

            chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
            rprint(f"Reactor Chi2: {chi2:.2f}")
            if params[2] == analysis_info["SIN12"]:
                rprint("Saving data to sin13")
                react_sin13_df.loc[params[0], params[1]] = chi2
            if params[1] == analysis_info["SIN13"]:
                rprint("Saving data to sin12")
                react_sin12_df.loc[params[0], params[2]] = chi2
            if (
                params[2] != analysis_info["SIN12"]
                and params[1] != analysis_info["SIN13"]
            ):
                rprint(f"[cyan]INFO: sin12 and sin13 do not match default[/cyan]")

            react_df.loc[i] = [params[0], params[1], params[2], chi2]

        # Delete files if they already exist
        path = f'{paths["signal_path"]}/results'
        rprint(f"\nSaving data to {path}")

        # Create the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        file_names = [
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl",
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl",
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl",
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl",
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_df.pkl",
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_df.pkl",
        ]

        for file_name in file_names:
            if os.path.exists(file_name):
                print(f"Removing {file_name}")
                os.remove(file_name)

        # Save the dataframes to pickle files
        solar_sin12_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl"
        )
        solar_sin13_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl"
        )
        react_sin12_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl"
        )
        react_sin13_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl"
        )
        solar_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_df.pkl"
        )
        react_df.to_pickle(
            f"{path}/{name}_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_df.pkl"
        )
