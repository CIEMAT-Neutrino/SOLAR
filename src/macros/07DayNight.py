import sys

sys.path.insert(0, "../../")

from lib import *

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
    "--folder", type=str, help="The name of the background folder", default="reduced"
)
parser.add_argument(
    "--signal_uncertanty",
    type=float,
    help="The signal uncertanty for the analysis",
    default=0.00,
)
parser.add_argument(
    "--background_uncertanty",
    type=float,
    help="The background uncertanty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy label for the analysis",
    default=["Cluster", "Total", "Selected", "Solar"],
)
parser.add_argument(
    "--fiducial",
    nargs="+",
    type=int,
    help="The fiducial cut for the analysis",
    default=np.arange(0, 140, 20),
)
parser.add_argument(
    "--exposure",
    nargs="+",
    type=int,
    help="The exposure array for the analysis",
    default=np.logspace(0, 2, 20),
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
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=10.0
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

config = args.config
name = args.name
folder = args.folder

energies = args.energy
fiducials = args.fiducial
hits = args.nhits
ophits = args.ophits
adjcls = args.adjcls

configs = {config: [name]}

user_input = {
    "exposure": args.exposure,
    "signal_uncertanty": args.signal_uncertanty,
    "background_uncertanty": args.background_uncertanty,
    "threshold": args.threshold,
    "rewrite": args.rewrite,
    "debug": args.debug,
}

# results = dict()

components = ["neutron", "gamma", "Solar"]
thld_idx = np.where(daynight_rebin_centers > user_input["threshold"])[0][0]
rprint(
    f"[INFO] Threshold {user_input['threshold']} found to correspond to index {thld_idx}"
)
for config in configs:
    for idx, name in enumerate(configs[config]):
        sigmas = []

        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/nominal/DAYNIGHT/{config}/{name}/{config}_{name}_rebin.pkl"
        )

        for bkg, bkg_label, color in [
            ("neutron", "neutron", "green"),
            ("gamma", "gamma", "black"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder}/DAYNIGHT/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
            )
            plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

        background = np.zeros(len(daynight_rebin_centers) - thld_idx, dtype=float)
        background_error = np.zeros(len(daynight_rebin_centers) - thld_idx, dtype=float)
        all_components = np.ones(len(daynight_rebin_centers) - thld_idx, dtype=bool)

        for energy_label in energies:
            last_sigma2 = 1e6
            last_sigma3 = 1e6
            for (idx, fiducial), (jdx, nhit), ophit, adjcl in track(
                product(
                    enumerate(fiducials),
                    enumerate(hits),
                    ophits,
                    adjcls,
                ),
                description=f"Looping over analysis cuts for {energy_label}Energy...",
                total=len(fiducials) * len(hits) * len(ophits) * len(adjcls),
            ):

                for array in [
                    background,
                    background_error,
                ]:
                    array.fill(0)

                all_components.fill(1)
                this_df = plot_df.loc[
                    (plot_df["EnergyLabel"] == energy_label)
                    * (plot_df["Fiducialized"] == int(fiducial))
                    * (plot_df["NHits"] == int(nhit))
                    * (plot_df["OpHits"] == int(ophit))
                    * (plot_df["AdjCl"] == int(adjcl))
                ]
                if this_df.empty:
                    print(
                        f"[WARNING] No data for {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                    )
                    continue

                for component in components:
                    comp_df = this_df.loc[(this_df["Component"] == component)]
                    if comp_df.empty:
                        print(
                            f"[WARNING] No data for {component} in {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                        )
                        continue
                    comp_df.fillna(0, inplace=True)

                    if component == "Solar":
                        this_comp_df = comp_df.loc[comp_df["Mean"] == "Day"]
                        signal_day = np.asarray(
                            this_comp_df["Counts"].values[0][thld_idx:]
                        )
                        all_components = all_components * (
                            np.asarray(comp_df["Counts"].values[0][thld_idx:]) > 0
                        )

                        s_error_day = this_comp_df["Error"].values[0][thld_idx:]

                        this_comp_df = comp_df.loc[comp_df["Mean"] == "Night"]
                        signal_night = np.asarray(
                            this_comp_df["Counts"].values[0][thld_idx:]
                        )

                    else:
                        background = (
                            background
                            + np.asarray(comp_df["Counts"].values[0])[thld_idx:]
                        )
                        background_error = (
                            background_error
                            + np.asarray(comp_df["Error"].values[0])[thld_idx:]
                        )
                        all_components = all_components * (
                            np.asarray(comp_df["Counts"].values[0])[thld_idx:] > 0
                        )

                found_sigma2 = False
                found_sigma3 = False
                sigma2, sigma3 = 0, 0
                gaussian_significances = []
                asimov_significances = []
                sigma2s = []
                sigma3s = []
                for factor in user_input["exposure"]:
                    # day_counts = factor * (signal_day + background / 2)
                    # night_counts = factor * (signal_night + background / 2)

                    # fitter = Asymmetry_Fitter(
                    #     N_day=np.asarray(day_counts),
                    #     N_night=np.asarray(night_counts),
                    #     B_hat=np.asarray(factor * background / 2),
                    #     sigma_B=np.asarray(
                    #         factor
                    #         * user_input["background_uncertanty"]
                    #         * background
                    #         / 2
                    #     ),
                    # )

                    # try:
                    #     chi2, B_fit, S_fit = fitter.Fit(
                    #         B_init=np.asarray(factor * background / 2),
                    #         S_day_init=np.asarray(signal_day),
                    #         S_night_init=np.asarray(signal_night),
                    #     )
                    # except ValueError:
                    #     print(
                    #         f"[WARNING] Fit failed for {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                    #     )
                    #     continue

                    gaussian_significance = evaluate_significance(
                        factor * all_components * (signal_night - signal_day),
                        factor * all_components * (background / 2 + signal_day),
                        # signal_uncertanty=(1 / ((factor * signal_day) ** 0.5)),
                        background_uncertanty=(
                            1 / ((factor * (background / 2 + signal_day)) ** 0.5)
                        ),
                    )
                    # Substitute nan values with 0
                    gaussian_significance = np.nan_to_num(gaussian_significance, nan=0)
                    gaussian_significance = (
                        np.sum(np.power(gaussian_significance, 2)) ** 0.5
                    )
                    gaussian_significances.append(gaussian_significance)

                    asimov_significance = evaluate_significance(
                        factor * all_components * (signal_night - signal_day),
                        factor * all_components * (background / 2 + signal_day),
                        # signal_uncertanty=(1 / ((factor * signal_day) ** 0.5)),
                        background_uncertanty=(
                            1 / ((factor * (background / 2 + signal_day)) ** 0.5)
                        ),
                        type="asimov",
                    )
                    # Substitute nan values with 0
                    asimov_significance = np.nan_to_num(asimov_significance, nan=0)
                    asimov_significance = (
                        np.sum(np.power(asimov_significance, 2)) ** 0.5
                    )
                    asimov_significances.append(asimov_significance)
                    # rprint(asimov_significance)
                    if asimov_significance > 2 and found_sigma2 == False:
                        sigma2 = factor
                        found_sigma2 = True
                        if sigma2 < last_sigma2:
                            rprint(
                                f"Found sigma2 with exposure {factor:.0f} for fiducial {fiducial} nhits {nhit} ophits {ophit} and adjcls {adjcl}"
                            )
                            last_sigma2 = sigma2

                    if asimov_significance > 3 and found_sigma3 == False:
                        sigma3 = factor
                        found_sigma3 = True
                        if sigma3 < last_sigma3:
                            rprint(
                                f"Found sigma3 with exposure {factor:.0f} for fiducial {fiducial} nhits {nhit} ophits {ophit} and adjcls {adjcl}"
                            )
                            last_sigma3 = sigma3

                    sigma2s.append(sigma2)
                    sigma3s.append(sigma3)

                sigmas.append(
                    {
                        "Config": config,
                        "Name": name,
                        "EnergyLabel": energy_label,
                        "Sigma2": sigma2s,
                        "Sigma3": sigma3s,
                        "Exposure": user_input["exposure"],
                        "Fiducialized": fiducial,
                        "NHits": nhit,
                        "OpHits": ophit,
                        "AdjCl": adjcl,
                        # "LogL": np.sqrt(chi2),
                        "Gaussian": gaussian_significances,
                        "Asimov": asimov_significances,
                    }
                )

        sigmas_df = pd.DataFrame(sigmas)
        save_df(
            sigmas_df,
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{folder}",
            config=config,
            name=name,
            filename=f"DayNight_Results",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
