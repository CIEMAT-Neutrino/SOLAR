import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/preselection/clustering"
data_path = f"{root}/output/data/preselection/clustering/"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

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
    "--name", type=str, help="The name of the configuration", default="marley_official"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "CORRECTION",
    "label": {
        "marley": "Neutrino",
        "neutron": "Neutron",
        "gamma": "Gamma",
        None: "Particle",
    },
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=["PRESELECTION"],
    signal = "marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

logy = False
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        nhit_list = []
        energy_scan_list = []

        bin_width = 0.05
        x_axis = np.arange(0, 1 + 2 * bin_width, bin_width)
        x_centers = x_axis[:-1] + bin_width / 2

        # --- fraction histograms per NHit (int pkl) ---
        for hit in nhits[:9]:
            this_reco_data, mask, output = compute_filtered_run(
                run,
                configs,
                params={("Reco", "NHits"): ("equal", hit)},
                debug=user_input["debug"],
            )
            this_reco_data = this_reco_data["Reco"]
            if len(this_reco_data["Charge"]) == 0:
                continue
            this_reco_data["ChargeFraction"] = (
                this_reco_data["Charge"] / this_reco_data["ElectronCharge"]
            )

            for _var, _arr in [
                ("Purity", this_reco_data["Purity"]),
                ("Completeness", this_reco_data["ChargeFraction"]),
            ]:
                y, _ = np.histogram(_arr, bins=x_axis)
                nhit_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "#Hits": hit,
                        "Values": 100 * x_centers,
                        "Counts": y,
                        "Density": y / (np.sum(y) * bin_width),
                        "Variable": _var,
                    }
                )

        # --- fraction histogram for all NHits combined (NaN row in int pkl) ---
        _reco = run["Reco"]
        _base_mask = (
            (_reco["Geometry"] == info["GEOMETRY"])
            & (_reco["Version"] == info["VERSION"])
            & (_reco["Name"] == name)
        )
        _all_purity = _reco["Purity"][_base_mask]
        _all_comp = _reco["Charge"][_base_mask] / _reco["ElectronCharge"][_base_mask]
        for _var, _arr in [("Purity", _all_purity), ("Completeness", _all_comp)]:
            y, _ = np.histogram(_arr, bins=x_axis)
            nhit_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "#Hits": None,
                    "Values": 100 * x_centers,
                    "Counts": y,
                    "Density": y / (np.sum(y) * bin_width),
                    "Variable": _var,
                }
            )

        # --- energy scan for all NHits + per NHit (NaN pkl) ---
        for hit in [None] + list(nhits[:9]):
            _hmask = _base_mask if hit is None else _base_mask & (_reco["NHits"] == hit)
            _es_energy, _es_purity, _es_purity_err = [], [], []
            _es_completeness, _es_completeness_err = [], []
            for energy in lowe_energy_centers:
                _emask = _hmask & (
                    (_reco["SignalParticleK"] >= energy - lowe_ebin / 2)
                    & (_reco["SignalParticleK"] < energy + lowe_ebin / 2)
                )
                _purity_vals = _reco["Purity"][_emask]
                _comp_vals = _reco["Charge"][_emask] / _reco["ElectronCharge"][_emask]
                if len(_purity_vals) < 10:
                    continue
                n = len(_purity_vals)
                _es_energy.append(energy)
                _es_purity.append(100 * np.mean(_purity_vals))
                _es_purity_err.append(100 * np.std(_purity_vals) / np.sqrt(n))
                _es_completeness.append(100 * np.mean(_comp_vals))
                _es_completeness_err.append(100 * np.std(_comp_vals) / np.sqrt(n))

            for _var, _vals, _errs in [
                ("Purity", _es_purity, _es_purity_err),
                ("Completeness", _es_completeness, _es_completeness_err),
            ]:
                energy_scan_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "#Hits": hit,
                        "Values": np.asarray(_es_energy, dtype=float),
                        "Counts": np.asarray(_vals),
                        "CountsError": np.asarray(_errs),
                        "Density": np.asarray(_vals),
                        "Variable": _var,
                    }
                )

        nhit_df = pd.DataFrame(nhit_list)
        energy_df = pd.DataFrame(energy_scan_list)

        for df_label in ["Purity", "Completeness"]:
            this_plot_df = nhit_df[nhit_df["Variable"] == df_label].copy()
            this_plot_df = explode(this_plot_df, ["Values", "Density", "Counts"])
            fig = px.line(
                this_plot_df,
                x="Values",
                y="Density",
                color="#Hits",
                line_shape="hvh",
                color_discrete_sequence=colors,
            )

            fig = format_coustom_plotly(
                fig,
                title=f"Cluster {df_label} - {config} {name}",
                tickformat=(".1f", None),
                log=(False, True),
                debug=user_input["debug"],
                legend_title="#Hits",
            )

            fig.update_layout(yaxis_title="Density")
            fig.update_layout(xaxis_title=f"{df_label} (%)")

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Cluster_{df_label}",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        save_df(
            nhit_df,
            data_path,
            config,
            name,
            filename="Clustering_Efficiency_NHit",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        save_df(
            energy_df,
            data_path,
            config,
            name,
            filename="Clustering_Efficiency_Energy",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
