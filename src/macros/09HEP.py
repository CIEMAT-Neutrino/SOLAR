import sys

sys.path.insert(0, "../../")

from lib import *

data_path = f"{root}/data/solar/"
save_path = f"{root}/images/hep"
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    default=0.3,
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
    default=np.logspace(0, np.log10(400), 20),
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
    "--mc_threshold",
    type=int,
    help="The mc count threshold for the analysis",
    default=100,
)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=16.0
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
configs = {args.config: [args.name]}
rprint(args)

user_input = {
    "exposure": args.exposure,
    "signal_uncertanty": args.signal_uncertanty,
    "background_uncertanty": args.background_uncertanty,
    "mc_threshold": args.mc_threshold,
    "threshold": args.threshold,
    "rewrite": args.rewrite,
    "debug": args.debug,
}

# results = dict()

components = ["neutron", "gamma", "8B", "hep"]
thld_idx = np.where(hep_rebin_centers > user_input["threshold"])[0][0]
rprint(
    f"[INFO] Threshold {user_input['threshold']} found to correspond to index {thld_idx}"
)
for config in configs:
    for jdx, name in enumerate(configs[config]):
        sigmas = []

        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/nominal/HEP/{config}/{name}/{config}_{name}_rebin.pkl"
        )

        for bkg, bkg_label, color in [
            ("neutron", "neutron", "green"),
            ("gamma", "gamma", "black"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder}/HEP/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
            )
            plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

        background = np.zeros(len(hep_rebin_centers) - thld_idx, dtype=float)
        background_error = np.zeros(
            (len(components) - 1, len(hep_rebin_centers) - thld_idx), dtype=float
        )
        signal_detection = np.zeros(len(hep_rebin_centers) - thld_idx, dtype=bool)

        for energy_label in args.energy:
            last_sigma2 = 1e6
            last_sigma3 = 1e6
            for (idx, fiducial), (jdx, nhit), ophit, adjcl in track(
                product(
                    enumerate(args.fiducial),
                    enumerate(args.nhits),
                    args.ophits,
                    args.adjcls,
                ),
                description=f"Looping over analysis cuts for {energy_label}Energy...",
                total=len(args.fiducial)
                * len(args.nhits)
                * len(args.ophits)
                * len(args.adjcls),
            ):

                for array in [background, background_error, signal_detection]:
                    array.fill(0)

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

                for idx, (component, osc, uncertanty_error) in enumerate(
                    zip(
                        components,
                        ["Truth", "Truth", "Osc", "Osc"],
                        [
                            user_input["background_uncertanty"],
                            user_input["background_uncertanty"],
                            user_input["signal_uncertanty"],
                            user_input["signal_uncertanty"],
                        ],
                    )
                ):
                    sufficient_mc_counts = True
                    comp_df = this_df.loc[(this_df["Component"] == component)]
                    if comp_df.empty:
                        rprint(
                            f"[WARNING] No data for {component} in {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                        )
                        continue
                    comp_df.fillna(0, inplace=True)
                    signal_detection.fill(0)

                    this_comp_df = comp_df.loc[
                        (comp_df["Oscillation"] == osc) * (comp_df["Mean"] == "Mean")
                    ]
                    if component == "hep":
                        sufficient_mc_counts = sufficient_mc_counts * (
                            np.sum(this_comp_df["MCCounts"].values[0][thld_idx:])
                            > user_input["mc_threshold"]
                        )
                        hep = this_comp_df["Counts"].values[0][thld_idx:]

                        s_error = np.divide(
                            this_comp_df["Error"].values[0][thld_idx:],
                            this_comp_df["Counts"].values[0][thld_idx:],
                            out=np.zeros_like(
                                this_comp_df["Counts"].values[0][thld_idx:]
                            ),
                            where=this_comp_df["Counts"].values[0][thld_idx:] != 0,
                        )

                    else:
                        sufficient_mc_counts = sufficient_mc_counts * (
                            np.sum(this_comp_df["MCCounts"].values[0][thld_idx:])
                            > user_input["mc_threshold"]
                        )
                        background = (
                            background + this_comp_df["Counts"].values[0][thld_idx:]
                        )

                        background_statistical = np.divide(
                            comp_df["Error"].values[0][thld_idx:],
                            comp_df["Counts"].values[0][thld_idx:],
                            out=np.zeros_like(comp_df["Counts"].values[0][thld_idx:]),
                            where=comp_df["Counts"].values[0][thld_idx:] != 0,
                        )
                        background_systematic = uncertanty_error * np.ones(
                            len(background_statistical)
                        )
                        background_error[idx] = (
                            np.sum(
                                (
                                    np.power(background_statistical, 2),
                                    np.power(background_systematic, 2),
                                ),
                                axis=0,
                            )
                            ** 0.5
                        )
                        background_error[idx] = np.multiply(
                            background_error[idx],
                            this_comp_df["Counts"].values[0][thld_idx:],
                            out=np.zeros_like(background_error[idx]),
                            where=this_comp_df["Counts"].values[0][thld_idx:] != 0,
                        )
                        # Substitute nan values with 0
                        background_error[idx] = np.nan_to_num(
                            background_error[idx],
                            nan=0,
                            posinf=0,
                            neginf=0,
                        )

                if not sufficient_mc_counts:
                    # print(
                    #     f"[WARNING] Not enough mc data for {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                    # )
                    continue

                combined_b_error = np.sum(background_error, axis=0)
                found_sigma2 = False
                found_sigma3 = False
                sigma2, sigma3 = 0, 0
                gaussian_significances = [[], [], []]
                asimov_significances = [[], [], []]
                sigma2s = []
                sigma3s = []
                for factor in user_input["exposure"]:
                    # Signal threshold to ensure 3 sigma detection given the signal's uncertanty
                    for kdx, detection_requirement in enumerate([2.9, 3, 3.1]):
                        signal_detection = (
                            factor
                            * np.asarray(hep)
                            * (
                                1
                                - detection_requirement
                                * user_input["signal_uncertanty"]
                            )
                            > 1
                        )
                        gaussian_significance = evaluate_significance(
                            signal_detection * factor * np.asarray(hep),
                            signal_detection * factor * np.asarray(background),
                            background_uncertanty=signal_detection
                            * factor
                            * np.asarray(combined_b_error),
                        )

                        gaussian_significance = (
                            np.sum(np.power(gaussian_significance, 2)) ** 0.5
                        )
                        gaussian_significances[kdx].append(gaussian_significance)

                        # Compute the asimov significance
                        asimov_significance = evaluate_significance(
                            signal_detection * factor * np.asarray(hep),
                            signal_detection * factor * np.asarray(background),
                            background_uncertanty=signal_detection
                            * factor
                            * np.asarray(combined_b_error),
                            type="asimov",
                        )

                        asimov_significance = (
                            np.sum(np.power(asimov_significance, 2)) ** 0.5
                        )
                        asimov_significances[kdx].append(asimov_significance)

                    if asimov_significances[1][-1] > 2 and found_sigma2 == False:
                        sigma2 = factor
                        found_sigma2 = True
                        if sigma2 < last_sigma2:
                            rprint(
                                f"Found sigma2 with exposure {factor:.0f} for fiducial {fiducial} nhits {nhit} ophits {ophit} and adjcls {adjcl}"
                            )
                            last_sigma2 = sigma2

                    if asimov_significances[1][-1] > 3 and found_sigma3 == False:
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
                        "Gaussian+Error": gaussian_significances[0],
                        "Gaussian": gaussian_significances[1],
                        "Gaussian-Error": gaussian_significances[2],
                        "Asimov+Error": asimov_significances[0],
                        "Asimov": asimov_significances[1],
                        "Asimov-Error": asimov_significances[2],
                    }
                )

        sigmas_df = pd.DataFrame(sigmas)
        save_df(
            sigmas_df,
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}",
            config,
            name,
            filename="HEP_Results",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
