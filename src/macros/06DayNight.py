import sys

sys.path.insert(0, "../../")

from lib import *

data_path = f"{root}/data/solar/"
save_path = f"{root}/images/solar/results/"
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
    "--rewrite", action="store_true", help="Rewrite the files", default=True
)
parser.add_argument(
    "--debug", action="store_true", help="Debug the files", default=True
)

config = parser.parse_args().config
name = parser.parse_args().name

configs = {config: [name]}

user_input = {
    "exposure": np.logspace(0, 2, 20),
    "s_uncertanty": 0.00,
    "b_uncertanty": 0.02,
    "threshold": 8,
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}

(dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
    path=f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/pkl/rebin/",
    ext="pkl",
    auto=False,
    debug=True,
)
results = dict()

for dm2, sin13, sin12 in zip(dm2_list, sin13_list, sin12_list):
    rprint(f"{dm2}, {sin13}, {sin12}")
    solar_df = process_oscillation_map(
        dm2_value=dm2, sin13_value=sin13, sin12_value=sin12, convolve=False, debug=False
    )
    solar_df = rebin_df(
        solar_df, show=False, convolve=False, save=False, save_path=None, debug=False
    )

nadir_data = get_nadir_angle(
    path="/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/",
    debug=user_input["debug"],
)
nadir = interpolate.interp1d(
    nadir_data[0], nadir_data[1], kind="linear", fill_value="extrapolate"
)

nadir_y = nadir(x=solar_df.index)
nadir_y = nadir_y / np.sum(nadir_y)

night_osc_prob = np.zeros(len(solar_df.columns))
day_osc_prob = np.zeros(len(solar_df.columns))
for idx, energy in enumerate(solar_df.columns):
    night_osc_prob[idx] = (
        solar_df.loc[-1:0][energy] * nadir_y[: int(len(nadir_y) / 2)]
    ).sum() / nadir_y[: int(len(nadir_y) / 2)].sum()
    day_osc_prob[idx] = (
        solar_df.loc[0:1][energy] * nadir_y[int(len(nadir_y) / 2) :]
    ).sum() / nadir_y[int(len(nadir_y) / 2) :].sum()

day_night = 100 * (night_osc_prob.sum() - day_osc_prob.sum()) / night_osc_prob.sum()
sk_day_night = (
    100
    * (day_osc_prob.sum() - night_osc_prob.sum())
    / (0.5 * (day_osc_prob.sum() + night_osc_prob.sum()))
)
asymmetry = (day_osc_prob - night_osc_prob) / (0.5 * (day_osc_prob + night_osc_prob))
# Interpolate the asymmetry to the energy bins
asymmetry = interpolate.interp1d(
    solar_df.columns, asymmetry, kind="linear", fill_value="extrapolate"
)
print(f"Day Night Asymmetry: {day_night:.3f}")
print(f"SK Day Night Asymmetry: {sk_day_night:.3f}")

thld_idx = np.where(rebin_centers > user_input["threshold"])[0][0]
for config in configs:
    for idx, name in enumerate(configs[config]):
        sigmas = []
        plot_data = {}
        fastest_sigma2, fastest_sigma3, highest_sigma = {}, {}, {}
        fastest_sigmas = [fastest_sigma2, fastest_sigma3]
        data_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{config}_{name}.pkl"
        )

        for bkg, bkg_label, color in [
            ("neutron", "neutron", "green"),
            ("gamma", "gamma", "black"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{config}_{bkg}.pkl"
            )
            data_df = pd.concat([data_df, bkg_df], ignore_index=True)

        # Rebin the data to have bigger energy bins
        data_df = rebin_df_columns(
            data_df, rebin, "Energy", "Counts", "Counts/Energy", "Error"
        )
        # Add an extra column significance with the same shape as the counts but filled with zeros
        data_df["Significance"] = np.zeros(len(data_df))
        plot_data[(config, name)] = data_df

        plot_df = explode(
            data_df,
            ["Counts", "Counts/Energy", "Error", "Energy"],
            debug=user_input["debug"],
        )
        plot_df["Counts"] = plot_df["Counts"].replace(0, np.nan)
        plot_df = plot_df[plot_df["Energy"] > user_input["threshold"]]
        plot_df["Counts/Energy"] = plot_df["Counts/Energy"].replace(0, np.nan)
        asymmetry_energy = asymmetry(rebin_centers[thld_idx:])
        components = ["neutron", "gamma", "Solar"]

        signal = np.zeros(len(rebin_centers) - thld_idx, dtype=float)
        s_error = np.zeros(len(rebin_centers) - thld_idx, dtype=float)
        background = np.zeros(len(rebin_centers) - thld_idx, dtype=float)
        b_error = np.zeros(len(rebin_centers) - thld_idx, dtype=float)
        all_components = np.ones(len(rebin_centers) - thld_idx, dtype=bool)

        for energy_label in ["Cluster", "Total", "Selected", "Solar"]:
            rprint(f"Energy Label: {energy_label}")
            for (idx, fiducial), (jdx, nhit), ophit, adjcl in track(
                product(
                    enumerate(np.arange(0, 140, 20)),
                    enumerate(nhits[:4]),
                    nhits[3:10],
                    nhits[::-1][6:],
                ),
                description="Looping over analysis cuts",
                total=len(np.arange(0, 140, 20))
                * len(nhits[:4])
                * len(nhits[3:10])
                * len(nhits[::-1][6:]),
            ):
                sigma2, sigma3 = 0, 0

                for array in [signal, s_error, background, b_error]:
                    array.fill(0)
                all_components.fill(1)

                this_df = plot_df[
                    (plot_df["EnergyLabel"] == energy_label)
                    * (plot_df["Fiducialized"] == fiducial)
                    * (plot_df["NHits"] == nhit)
                    * (plot_df["OpHits"] == ophit)
                    * (plot_df["AdjCl"] == adjcl)
                ]
                if this_df.empty:
                    print(
                        f"[WARNING] No data for {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                    )
                    continue

                for component in components:
                    comp_df = this_df[(this_df["Component"] == component)]
                    if comp_df.empty:
                        print(
                            f"[WARNING] No data for {component} in {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl"
                        )
                        continue
                    comp_df.fillna(0, inplace=True)

                    if component != "Solar":
                        background = background + np.asarray(comp_df["Counts"].values)
                        rel_b_error = np.divide(
                            np.asarray(comp_df["Error"].values),
                            np.asarray(comp_df["Counts"].values),
                            out=np.zeros_like(comp_df["Counts"].values),
                            where=comp_df["Counts"].values != 0,
                        )
                        b_error = np.power(
                            np.power(b_error, 2) + np.power(rel_b_error, 2), 0.5
                        )
                        all_components = all_components * (
                            np.asarray(comp_df["Counts"].values) != 0
                        )
                        # if np.sum(np.asarray(comp_df["Counts"].values) == 0) > 0:
                        # print(f"[WARNING] Some bins are empty for {component} in {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl")

                    else:
                        signal = signal + np.asarray(comp_df["Counts"].values)
                        s_error = s_error + np.divide(
                            np.asarray(comp_df["Error"].values),
                            np.asarray(comp_df["Counts"].values),
                            out=np.zeros_like(comp_df["Counts"].values),
                            where=comp_df["Counts"].values != 0,
                        )
                        all_components = all_components * (
                            np.asarray(comp_df["Counts"].values) != 0
                        )
                        # if np.sum(np.asarray(comp_df["Counts"].values) == 0) > 0:
                        # print(f"[WARNING] Some bins are empty for {component} in {energy_label} with {fiducial} fiducialized, {nhit} nhits and {ophit} ophits {adjcl} adjcl")

                found_sigma2 = False
                found_sigma3 = False
                for factor in user_input["exposure"]:
                    sigma2, sigma3 = 0, 0
                    day_counts = (
                        factor * (signal * (1 + asymmetry_energy) + background) / 2
                    )
                    night_counts = (
                        factor * (signal * (1 - asymmetry_energy) + background) / 2
                    )

                    denominator = (
                        day_counts
                        + night_counts
                        + (
                            factor
                            * (s_error + user_input["s_uncertanty"])
                            * signal
                            * (1 + asymmetry_energy)
                        )
                        ** 2
                        + (
                            factor
                            * (s_error + user_input["s_uncertanty"])
                            * signal
                            * (1 - asymmetry_energy)
                        )
                        ** 2
                        + 2
                        * (factor * (b_error + user_input["b_uncertanty"]) * background)
                    )
                    significance = all_components * np.divide(
                        (day_counts - night_counts),
                        np.power(denominator, 0.5),
                        out=np.zeros_like(denominator),
                        where=denominator != 0,
                    )

                    significance = np.sum(np.power(significance, 2)) ** 0.5

                    if significance > 2 and found_sigma2 == False:
                        sigma2 = factor
                        found_sigma2 = True
                        # Find entry in data_df and update the significance
                        data_df.loc[
                            (data_df["Name"] == name)
                            * (data_df["Component"] == "Solar")
                            * (data_df["Type"] == "signal")
                            * (data_df["EnergyLabel"] == energy_label)
                            * (data_df["Fiducialized"] == fiducial)
                            * (data_df["NHits"] == nhit)
                            * (data_df["OpHits"] == ophit)
                            * (data_df["AdjCl"] == adjcl),
                            "Significance",
                        ] = significance

                    if significance > 3 and found_sigma3 == False:
                        sigma3 = factor
                        found_sigma3 = True
                        # Find entry in data_df and update the significance
                        data_df.loc[
                            (data_df["Name"] == name)
                            * (data_df["Component"] == "Solar")
                            * (data_df["Type"] == "signal")
                            * (data_df["EnergyLabel"] == energy_label)
                            * (data_df["Fiducialized"] == fiducial)
                            * (data_df["NHits"] == nhit)
                            * (data_df["OpHits"] == ophit)
                            * (data_df["AdjCl"] == adjcl),
                            "Significance",
                        ] = significance

                    sigmas.append(
                        {
                            "Config": config,
                            "Name": name,
                            "EnergyLabel": energy_label,
                            "Sigma2": 0.5 * int(sigma2),
                            "Sigma3": 0.5 * int(sigma3),
                            "Values": significance,
                            "Exposure": factor,
                            "Fiducialized": fiducial,
                            "NHits": nhit,
                            "OpHits": ophit,
                            "AdjCl": adjcl,
                        }
                    )

        sigmas_df = pd.DataFrame(sigmas)
        for idx, sigma_label in enumerate(["Sigma2", "Sigma3"]):
            print(f"Evaluating {sigma_label} for {energy_label}")
            # Find the entry with the highest significance (max this_sigma_df["sigma_label"])
            this_sigma_df = sigmas_df[
                (sigmas_df["Config"] == config)
                * (sigmas_df["Name"] == name)
                * (sigmas_df["EnergyLabel"] == energy_label)
            ]
            if idx == 0:
                this_sigma = this_sigma_df[
                    this_sigma_df["Values"] == this_sigma_df["Values"].max()
                ]
                if len(this_sigma) >= 1:
                    # display(this_sigma)
                    this_sigma = this_sigma.iloc[0]
                elif len(this_sigma) == 0:
                    continue
                highest_sigma[(config, name, energy_label)] = {
                    "Sigma": float(this_sigma[sigma_label]),
                    "Values": float(this_sigma["Values"]),
                    "Fiducialized": int(this_sigma["Fiducialized"]),
                    "NHits": int(this_sigma["NHits"]),
                    "OpHits": int(this_sigma["OpHits"]),
                    "AdjCl": int(this_sigma["AdjCl"]),
                }

            # Find the entry with the fastest sigma (min this_sigma_df["sigma_label"])
            this_sigma_df = this_sigma_df[(this_sigma_df[sigma_label] > 0)]
            this_sigma = this_sigma_df[
                this_sigma_df[sigma_label] == this_sigma_df[sigma_label].min()
            ]
            this_sigma = this_sigma_df[
                this_sigma_df["Values"] == this_sigma_df["Values"].max()
            ]
            if len(this_sigma) >= 1:
                # display(this_sigma)
                this_sigma = this_sigma.iloc[0]
            elif len(this_sigma) == 0:
                continue
            fastest_sigmas[idx][(config, name, energy_label)] = {
                "Sigma": float(this_sigma[sigma_label]),
                "Values": float(this_sigma["Values"]),
                "Fiducialized": int(this_sigma["Fiducialized"]),
                "NHits": int(this_sigma["NHits"]),
                "OpHits": int(this_sigma["OpHits"]),
                "AdjCl": int(this_sigma["AdjCl"]),
            }

        for sigma_results, sigma_results_label in zip(
            [highest_sigma, fastest_sigma2, fastest_sigma3],
            ["highest", "fastest_sigma2", "fastest_sigma3"],
        ):
            save_df(
                pd.DataFrame(sigma_results),
                data_path,
                config,
                name,
                sigma_results_label,
                user_input["rewrite"],
                debug=user_input["debug"],
            )
