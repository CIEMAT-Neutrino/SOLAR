import sys, json

sys.path.insert(0, "../../")
from lib import *

from lib.root_functions import Fitter
from lib.osc_functions import get_oscillation_datafiles

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
    "--uncertanty", type=float, help="The threshold for the analysis", default=0.02
)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=6.0
)
parser.add_argument(
    "--rewrite", action="store_true", help="Rewrite the files", default=True
)
parser.add_argument(
    "--debug", action="store_true", help="Debug the files", default=True
)

config = parser.parse_args().config
name = parser.parse_args().name
uncertanty = parser.parse_args().uncertanty
configs = {config: [name]}
# Find index where rebin_centers is above threshold
threshold = parser.parse_args().threshold
thld = np.where(rebin_centers > threshold)[0][0]


for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))

    for name in configs[config]:
        user_config = {
            "signal_path": f"{info['PATH']}/sensitivity/rebin/{config}/{name}",
            "background_path": f"{info['PATH']}/sensitivity/rebin/{config}/background",
        }

        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            dm2=None,
            sin13=[0.021],
            # sin13=None,
            sin12=None,
            path=f"{user_config['signal_path']}/",
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
            f'{user_config["signal_path"]}/{config}_{name}_ClusterEnergy_dm2_{analysis_info["SOLAR_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
        )
        pred2_df = pd.read_pickle(
            f'{user_config["signal_path"]}/{config}_{name}_ClusterEnergy_dm2_{analysis_info["REACT_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
        )
        bkg_df = pd.read_pickle(
            f'{user_config["background_path"]}/{config}_background_ClusterEnergy.pkl'
        )

        fake_df_dict = {}
        for dm2, sin13, sin12 in zip(dm2_list, sin13_list, sin12_list):
            fake_df_dict[(dm2, sin13, sin12)] = (
                pd.read_pickle(
                    f'{user_config["signal_path"]}/{config}_{name}_ClusterEnergy_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl'
                )
                + bkg_df
            )

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
            fitter = Fitter(
                obs_df[:, thld:],
                pred1_df[:, thld:],
                bkg_df[:, thld:],
                DayNight=True,
                SigmaPred=uncertanty,
                SigmaBkg=uncertanty,
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

            fitter = Fitter(
                obs_df[:, thld:],
                pred2_df[:, thld:],
                bkg_df[:, thld:],
                DayNight=True,
                SigmaPred=uncertanty,
                SigmaBkg=uncertanty,
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
        path = f'{user_config["signal_path"]}/results'
        rprint(f"\nSaving data to {path}")

        # Create the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        file_names = [
            f"{path}/{name}_solar_sin12_df.pkl",
            f"{path}/{name}_solar_sin13_df.pkl",
            f"{path}/{name}_react_sin12_df.pkl",
            f"{path}/{name}_react_sin13_df.pkl",
            f"{path}/{name}_solar_df.pkl",
            f"{path}/{name}_react_df.pkl",
        ]

        for file_name in file_names:
            if os.path.exists(file_name):
                print(f"Removing {file_name}")
                os.remove(file_name)

        # Save the dataframes to pickle files
        solar_sin12_df.to_pickle(f"{path}/{name}_solar_sin12_df.pkl")
        solar_sin13_df.to_pickle(f"{path}/{name}_solar_sin13_df.pkl")
        react_sin12_df.to_pickle(f"{path}/{name}_react_sin12_df.pkl")
        react_sin13_df.to_pickle(f"{path}/{name}_react_sin13_df.pkl")
        solar_df.to_pickle(f"{path}/{name}_solar_df.pkl")
        react_df.to_pickle(f"{path}/{name}_react_df.pkl")
