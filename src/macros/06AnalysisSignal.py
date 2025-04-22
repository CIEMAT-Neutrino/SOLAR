import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/solar/results/nominal"
data_path = f"{root}/data/solar/results/nominal"

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
    "--fiducial", type=int, help="The fiducial cut for the analysis", default=20
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=1
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=4
)
parser.add_argument(
    "--adjcl", type=int, help="The adjacent cluster cut for the analysis", default=10
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

config = parser.parse_args().config
name = parser.parse_args().name

ref_fiducial = parser.parse_args().fiducial
ref_nhits = parser.parse_args().nhits
ref_ophits = parser.parse_args().ophits
ref_adjcl = parser.parse_args().adjcl

configs = {config: [name]}

user_input = {
    "directory": {
        "marley": "signal/nominal",
    },
    "weights": {
        "marley": [
            "SignalParticleWeight",
            "SignalParticleWeightb8",
            "SignalParticleWeighthep",
        ],
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
    },
    "yzoom": {"marley": [0, 6]},
    "workflow": "ANALYSIS",
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
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

run, output, this_new_branches = compute_particle_weights(
    run,
    configs,
    params={
        "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
        "DEFAULT_SIGNAL_AZIMUTH": ["mean", "day", "night"],
    },
    rm_branches=True,
    output=output,
    debug=user_input["debug"],
)
rprint(f"{output}\nNew branches: {this_new_branches}")

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    plot_list = []
    for name, energy in product(
        configs[config],
        ["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    ):
        fig = make_subplots(rows=1, cols=1, subplot_titles=([energy]))

        save_pkl(
            run["Reco"]["SignalParticleK"],
            f"{data_path}",
            config,
            name,
            filename=f"AnalysisEnergy_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        save_pkl(
            run["Reco"][energy],
            f"{data_path}",
            config,
            name,
            filename=f"AnalysisData_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        save_pkl(
            run["Reco"]["SignalParticleWeight"],
            f"{data_path}",
            config,
            name,
            filename=f"AnalysisWeights_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        for (
            (idx, this_fiducial),
            this_nhit,
            this_ophit,
            this_adjcl,
            (weight, weight_labels, color),
        ) in track(
            product(
                enumerate(np.arange(0.00, 140, 20)),
                nhits[:10],
                nhits[3:10],
                nhits[::-1][10:],
                zip(
                    user_input["weights"][name],
                    user_input["weight_labels"][name],
                    user_input["colors"][name],
                ),
            ),
            total=6 * 10 * 7 * 10 * 3,
            description=f"Iterating over cut configurations for reco {energy}...",
        ):
            if this_fiducial == 0:
                mask = (run["Reco"]["NHits"] > this_nhit - 1) * (
                    run["Reco"]["AdjClNum"] < this_adjcl
                )

            else:
                mask = (
                    (run["Reco"]["NHits"] > this_nhit - 1)
                    * (run["Reco"]["AdjClNum"] < this_adjcl)
                    * (run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1)
                    * (np.absolute(run["Reco"]["RecoX"]) > this_fiducial * 0.1)
                    * (np.absolute(run["Reco"]["RecoX"]) < detector_x / 2)
                    * (np.absolute(run["Reco"]["RecoY"]) > 0)
                    * (
                        np.absolute(run["Reco"]["RecoY"])
                        < detector_y / 2 - this_fiducial
                    )
                    * (
                        np.absolute(run["Reco"]["RecoZ"])
                        > this_fiducial - info["DETECTOR_GAP_Z"]
                    )
                    * (
                        np.absolute(run["Reco"]["RecoZ"])
                        < info["DETECTOR_SIZE_Z"]
                        + info["DETECTOR_GAP_Z"]
                        - this_fiducial
                    )
                )

            idx_mask = np.where(mask == True)
            h, bins = np.histogram(run["Reco"][energy][idx_mask], bins=energy_edges)
            mc_filter = h > 1
            mc_counts = h.copy()
            h_rel_error = np.sqrt(h) / h

            for osc, mean, mean_label in zip(
                ["Truth", "Osc", "Osc", "Osc"],
                ["Mean", "Day", "Night", "Mean"],
                ["", "OscDay", "OscNight", "OscMean"],
            ):
                h, bins = np.histogram(
                    run["Reco"][energy][idx_mask],
                    bins=energy_edges,
                    weights=run["Reco"][f"{weight}{mean_label}"][idx_mask],
                )
                h *= mc_filter
                h_error = h * h_rel_error
                h_error[np.isnan(h_error)] = 0

                counts = np.sum(h[energy_centers > 10])
                plot_list.append(
                    {
                        "Idx": 0,
                        "Name": name,
                        "Component": weight_labels,
                        "Oscillation": osc,
                        "Mean": mean,
                        "Type": "signal",
                        "MCCounts": mc_counts.tolist(),
                        "Counts": h.tolist(),
                        "Energy": energy_centers,
                        "Error": h_error,
                        "EnergyLabel": energy.split("Energy")[0],
                        "Color": color,
                        "Fiducialized": int(this_fiducial),
                        "NHits": this_nhit,
                        "OpHits": this_ophit,
                        "AdjCl": this_adjcl,
                    }
                )

            if (
                this_nhit == ref_nhits
                and this_ophit == ref_ophits
                and this_adjcl == ref_adjcl
                and weight == "SignalParticleWeight"
            ):
                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=h,
                        mode="lines",
                        showlegend=True,
                        name=f"{this_fiducial/100:.1f}: {counts:.1e} counts",
                        line_shape="hvh",
                        line=dict(color=colors[1 + idx]),
                    ),
                    row=1,
                    col=1,
                )
                if this_fiducial == ref_fiducial:
                    # Save the energy spectrum to a file for further analysis
                    save_pkl(
                        np.asarray(mask),
                        f"{data_path}",
                        config,
                        name,
                        filename=f"AnalysisMask_{energy}_Fiducial{this_fiducial}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"]["SignalParticleK"][idx_mask],
                        f"{data_path}",
                        config,
                        name,
                        filename=f"AnalysisEnergy_{energy}_Fiducial{this_fiducial}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"][energy][idx_mask],
                        f"{data_path}",
                        config,
                        name,
                        filename=f"AnalysisData_{energy}_Fiducial{this_fiducial}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"][weight][idx_mask],
                        f"{data_path}",
                        config,
                        name,
                        filename=f"AnalysisWeights_{energy}_Fiducial{this_fiducial}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )

        # Add verticlal lines
        fig.add_vline(
            10,
            line_width=1,
            line_dash="dash",
            line_color="grey",
            annotation_text=" Threshold",
            annotation_position="bottom right",
        )

        fig = format_coustom_plotly(
            fig,
            title=f"Weighted {energy} ({name})",
            log=(False, True),
            ranges=(None, user_input["yzoom"][name]),
            legend=dict(x=0.7, y=0.99),
            legend_title="Fiducial (m)",
            tickformat=(".0f", ".0e"),
        )

        fig.update_xaxes(title="Reconstructed K.E. (MeV)")
        fig.update_yaxes(title="Counts per (kT · year)")
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Particle_{energy}_Fiducial_Scan_Hist",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    df = pd.DataFrame(plot_list)
    save_df(
        df,
        f"{info['PATH']}/{user_input['directory'][name]}",
        config=config,
        name=None,
        filename=f"{name}",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )

    for rebin, analysis in zip(
        [daynight_rebin, sensitivity_rebin, hep_rebin],
        ["DayNight", "Sensitivity", "HEP"],
    ):
        rebin_df = rebin_df_columns(
            df, rebin, "Energy", "Counts", "Counts/Energy", "Error"
        )

        save_df(
            rebin_df,
            f"{info['PATH']}/{user_input['directory'][name]}/{analysis.upper()}",
            config=config,
            name=name,
            filename=f"rebin",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
