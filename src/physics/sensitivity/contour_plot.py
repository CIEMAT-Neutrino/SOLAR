import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--signal", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    default="Nominal",
    choices=["Reduced", "Truncated", "Nominal"],
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK", "MainK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis in years.",
    default=30.0,
)
parser.add_argument(
    "--secondary_exposure", "--secondary-exposure",
    type=float, default=10.0,
    help=(
        "Also export the tagged secondary-exposure grids written by 06_significance.py "
        "(e.g. '_10Y') as a separate Sensitivity_<N>Y_Contours.pkl. 0/negative disables."
    ),
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--nuisance_profile",
    type=str,
    help="Nuisance parameter profile name (key in NUISANCE_PROFILES in config/analysis/config.json). Defaults to DEFAULT_NUISANCE_PROFILE.",
    default=None,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--study_label", type=str, default=None, help="Tag appended to image subdirectory to isolate study outputs.")
parser.add_argument("--charge_threshold", type=float, default=0, help="Charge threshold Q (ADC). When >0, reads chi2 grids from labeled template subfolders.")
parser.add_argument(
    "--draft",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Draft mode flag (passed through from pipeline). "
        "Contour plots will be generated from draft-mode chi2 grids if available."
    ),
)

args = parser.parse_args()
_ctx = study_context(args)
_study_suffix    = _ctx.study_suffix
_template_suffix = _ctx.template_suffix
_save_subfolder  = _ctx.save_subfolder
configs = {args.config: [args.signal]}
if args.debug:
    rprint(args)


def _load_best_cut_map(info: dict, args):
    _suffix = f"_{args.study_label}" if getattr(args, 'study_label', None) else ""
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    tried = []
    
    # Standard location: {PATH}/SENSITIVITY/{config}/{name}/{folder}/...
    for analysis in candidates if not _suffix else ["SENSITIVITY"]:
        for suffix in ([_suffix] if _suffix else [""]):
            filepath = (
                f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/"
                f"{args.config}_{args.signal}_highest_{analysis}{suffix}.pkl"
            )
            tried.append(filepath)
            if os.path.exists(filepath):
                if args.debug:
                    rprint(f"Loading best-cut map from: {filepath}")
                return pickle.load(open(filepath, "rb"))

    if args.debug:
        rprint("Unable to load any best-cut map. Checked:\n" + "\n".join(tried))
    return None

sensitivity = []
save_path = f"{root}/output/images/analysis/sensitivity"
# Tidy contour rows for the external plotting library (one row per grid).
_contour_rows: dict = {}   # exposure tag -> list of rows

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = load_analysis_info(str(root))
    _nuisance_profiles = analysis_info.get("NUISANCE_PROFILES", {})
    _default_profile   = analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
    profile_name       = args.nuisance_profile or _default_profile or "full"
    energy = args.energy
    nhits, adjcl, ophits = -1, -1, -1

    fastest_sigma = {(args.config, args.signal, args.energy): None}
    if args.nhits is None or args.adjcls is None or args.ophits is None:
        loaded = _load_best_cut_map(info, args)
        if loaded is not None:
            fastest_sigma = loaded
        elif args.study_label:
            raise FileNotFoundError(
                f"Missing best-cut map for study '{args.study_label}'; refusing to plot nominal cuts"
            )
        else:
            fastest_sigma = {
                (args.config, args.signal, args.energy): {
                    "NHits": 4,
                    "AdjCl": 10,
                    "OpHits": 4,
                }
            }
            if args.debug:
                rprint("Falling back to default cuts NHits4 AdjCl10 OpHits4")

    cut_keys = (
        list(fastest_sigma.keys())
        if args.nhits is None or args.adjcls is None or args.ophits is None
        else [(args.config, args.signal, args.energy)]
    )

    for name, key in product(configs[config], cut_keys):
        if args.energy is not None:
            energy = args.energy
        else:
            energy = key[2]

        if args.background:
            data_path = f"{info['PATH']}/SENSITIVITY/{config}/{args.signal}/{args.folder.lower()}/{energy}{_template_suffix}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%"
        else:
            data_path = f"{info['PATH']}/SENSITIVITY/{config}/{args.signal}/{args.folder.lower()}/{energy}{_template_suffix}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_only"

        invalid_marker = f"{data_path}/{name}_{energy}_Sensitivity_INVALID.json"
        if os.path.exists(invalid_marker):
            raise ValueError(
                f"Sensitivity results are marked invalid: {invalid_marker}. "
                "Regenerate 06_significance.py output before plotting."
            )

        nhits = int(args.nhits) if args.nhits is not None else int((fastest_sigma.get(key) or {}).get("NHits", 4))
        adjcl = int(args.adjcls) if args.adjcls is not None else int((fastest_sigma.get(key) or {}).get("AdjCl", 10))
        ophits = int(args.ophits) if args.ophits is not None else int((fastest_sigma.get(key) or {}).get("OpHits", 4))

        _stem = f"{data_path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"

        def _read_grids(tag):
            return {
                ("solar", "sin13"): pd.read_pickle(f"{_stem}_solar_sin13_df{tag}.pkl"),
                ("solar", "sin12"): pd.read_pickle(f"{_stem}_solar_sin12_df{tag}.pkl"),
                ("react", "sin13"): pd.read_pickle(f"{_stem}_react_sin13_df{tag}.pkl"),
                ("react", "sin12"): pd.read_pickle(f"{_stem}_react_sin12_df{tag}.pkl"),
            }

        # Secondary-exposure grids are written by 06_significance.py in the same pass;
        # export them under their own tag so the external library gets one file per exposure.
        _tags = [("", float(getattr(args, "exposure", np.nan)))]
        _sec = getattr(args, "secondary_exposure", None)
        if _sec and _sec > 0:
            _sec_tag = f"_{_sec:g}Y"
            if os.path.exists(f"{_stem}_solar_sin12_df{_sec_tag}.pkl"):
                _tags.append((_sec_tag, float(_sec)))

        for _tag, _tag_exposure in _tags:
            _grids = _read_grids(_tag)
            for (_ref, _par), _grid in _grids.items():
                _g = _grid.astype(float)
                _contour_rows.setdefault(_tag, []).append({
                    "Config": config, "Name": name, "Analysis": "Sensitivity",
                    "EnergyLabel": energy, "Study": args.study_label or "default",
                    "Label": _ref, "Variable": _par,
                    "Dm2": np.asarray(_g.index, dtype=float).tolist(),
                    "Values": np.asarray(_g.columns, dtype=float).tolist(),
                    "Significance": np.asarray(_g.to_numpy(dtype=float)).tolist(),
                    "SignificanceUnit": r"\Delta\chi^2",
                    "NuisanceProfile": profile_name,
                    "SignalUncertainty": float(args.signal_uncertainty),
                    "BackgroundUncertainty": float(args.background_uncertainty) if args.background else 0.0,
                    "NHits": int(nhits), "AdjCl": int(adjcl), "OpHits": int(ophits),
                    "Exposure": _tag_exposure,
                    "SourcePath": data_path,
                })

        # The figures below are made from the PRIMARY exposure grids.
        _primary = _read_grids("")
        solar_sin13_df = _primary[("solar", "sin13")].sort_index().sort_index(axis=1)
        solar_sin12_df = _primary[("solar", "sin12")].sort_index().sort_index(axis=1)
        react_sin13_df = _primary[("react", "sin13")].sort_index().sort_index(axis=1)
        react_sin12_df = _primary[("react", "sin12")].sort_index().sort_index(axis=1)
        
        contours = np.arange(0, 4, 1)
        for df, df_name in zip([solar_sin13_df, solar_sin12_df, react_sin13_df, react_sin12_df], ["solar_sin13_df", "solar_sin12_df", "react_sin13_df", "react_sin12_df"]):
            # IMPORTANT: Compute chi2_min BEFORE replacing zeros with NaN
            chi2_values_original = df.values.astype(float)
            finite_mask_original = np.isfinite(chi2_values_original)
            if np.any(finite_mask_original):
                chi2_min = float(np.min(chi2_values_original[finite_mask_original]))
            else:
                chi2_min = 0.0
            
            # Now replace negative values and zeros with NaN for display
            df[df < 0] = 0.0
            df.replace(0, np.nan, inplace=True)
            
            # Compute delta chi2 using the original chi2_min
            chi2_values = df.values.astype(float)
            finite_mask = np.isfinite(chi2_values)
            
            delta_chi2 = chi2_values - chi2_min
            delta_chi2 = np.maximum(delta_chi2, 0.0)
            delta_chi2[~finite_mask] = np.nan

            sensitivity.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Label": df_name.split("_")[0],
                    "Variable": df_name.split("_")[1],
                    "Dm2": df.index.astype(float).values,
                    "Values": df.columns.astype(float).values,
                    "Significance": np.sqrt(delta_chi2).tolist(),
                    "Chi2Min": chi2_min,
                }
            )

            # Modify ylgnbu_r coloraxis to have last color with white
            colorscale = [[0, "navy"], [0.5, "teal"], [1, "white"]]
            fig = make_subplots(
                1,
                1,
                subplot_titles=(
                    [
                        f"{energy}, min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}"
                    ]
                ),
            )
            fig.add_trace(
                go.Contour(
                    x=df.columns.astype(float),
                    y=df.index,
                    z=np.sqrt(delta_chi2),
                    connectgaps=True,
                    coloraxis="coloraxis",
                    contours=dict(start=0, end=contours[-1], size=1),
                    name="DUNE",
                    showlegend=True,
                )
            )

            fig = format_coustom_plotly(
                fig,
                # title=f"DUNE Contours for Solar Best Fit ({unicode('Delta')}m²{subscript(21)} {6e-5:.0e} eV²)",
                title=f"{config} {name} {energy}",
                tickformat=(".2f", ".0e"),
                add_watermark=True,
            )

            fig.update_coloraxes(
                colorbar_title=f"{unicode('sigma')}",
            )
            if "sin12" in df_name:
                fig.update_layout(
                    coloraxis=dict(colorscale=colorscale),
                    xaxis=dict(range=[0.15, 0.45]),
                    yaxis=dict(range=[3e-5, 1e-4]),
                )
            else:
                fig.update_layout(
                    coloraxis=dict(colorscale=colorscale),
                    xaxis=dict(range=[0.01, 0.04]),
                    yaxis=dict(range=[3e-5, 1e-4]),
                )

            # Add an ellipse at position y=7.4 and x=0.303
            if "sin12" in df_name:
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=0.304,
                    y0=7.455e-5,
                    x1=0.312,
                    y1=7.595e-5,
                    # opacity=0.2,
                    fillcolor="black",
                    line_color="black",
                    # showlegend=True,
                    # name=f"(JUNO) 3{unicode('sigma')}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1.5],
                        y=[0.75],
                        name=f"JUNO (3{unicode('sigma')})",
                        text=f"JUNO (3{unicode('sigma')})",
                        mode="markers",
                        marker=dict(size=12, color="black"),
                    )
                )
                sno_paths = ["contour1_tan.csv", "contour2_tan.csv", "contour3_tan.csv"]
                solar_paths = ["contour1.csv", "contour2.csv", "contour3.csv"]
                kamland_paths = ["contour1.csv", "contour2.csv", "contour3.csv"]

                for file_paths, label, color in zip(
                    [solar_paths, kamland_paths],
                    ["Solar", "KamLAND"],
                    ["blue", "grey"],
                ):
                    dash_list = ["solid", "dot", "dash"]
                    for idx, file_path in enumerate(file_paths):
                        compute_sin = False
                        deltam_factor = 1e-5
                        folder_path = f"{root}/external/contours/{label}"
                        # Check if the file exists
                        if not os.path.exists(f"{folder_path}/{file_path}"):
                            print(f"File {folder_path}/{file_path} does not exist.")
                            sys.exit(1)

                        file_name = file_path.split(".")[0]

                        if file_path.split(".")[-1] != "csv":
                            print(f"File {file_path} is not a CSV file.")
                            sys.exit(1)

                        if file_name.split("_")[-1] == "tan":
                            compute_sin = True
                            deltam_factor = 1e-4

                        # Load the CSV file
                        data = load_contour_csv(
                            f"{folder_path}/{file_path}",
                            compute_sin=compute_sin,
                            deltam_factor=deltam_factor,
                        )

                        # Draw the contour
                        fig = draw_contour(fig, idx, label, data, color, dash_list[idx])

            # Show legend inside the plot
            fig.update_layout(
                legend=dict(
                    title="Contours",
                    orientation="v",
                    font=dict(size=18),
                    bgcolor="rgba(255,255,255,0.7)",
                )
            )

            if args.background:
                figure_name = f"{df_name}_{args.folder}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertainty:.0f}_Bkg{100*args.background_uncertainty:.0f}"
            else:
                figure_name = f"{df_name}_{args.folder}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertainty:.0f}"

            fig.update_xaxes(title=f"sin²{unicode('theta')}{subscript(12)}" if "sin12" in df_name else f"sin²{unicode('theta')}{subscript(13)}")
            fig.update_yaxes(title=f"{unicode('Delta')}m²{subscript(21)} (eV²)")
            save_figure(
                fig,
                save_path,
                config=config,
                name=name,
                subfolder=f"{_save_subfolder}/{profile_name}",
                filename=figure_name,
                rm=args.rewrite,
                debug=args.plot,
            )
    
    save_pkl(
        pd.DataFrame(sensitivity),
        f"{analysis_info['PATH']}/SENSITIVITY",
        config=config,
        name=name,
        subfolder=_save_subfolder,
        filename=f"Sensitivity_{energy}" if args.nhits is None and args.adjcls is None and args.ophits is None else f"Sensitivity_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.debug,
    )


# ── Contours DataFrame ─────────────────────────────────────────────────────────
# Published alongside the figures so the external plotting library can build its own
# comparisons; same stem convention as Sensitivity_Counts / Sensitivity_Significance.
if _contour_rows:
    for _tag, _rows in _contour_rows.items():
        _contours_df = pd.DataFrame(_rows)
        _fname = "Sensitivity_Contours" if not _tag else f"Sensitivity{_tag}_Contours"
        for _cfg, _grp in _contours_df.groupby("Config"):
            for _nm, _sub in _grp.groupby("Name"):
                _payload = _sub.reset_index(drop=True)
                for _dest in [
                    f"{json.loads(open(f'{root}/config/{_cfg}/{_cfg}_config.json').read())['PATH']}/SENSITIVITY",
                    f"{root}/output/data/analysis/sensitivity",
                ]:
                    # one invocation per nuisance profile -> upsert, else the last profile
                    # silently erases the others
                    _merged = upsert_df_rows(
                        _payload, _dest, config=_cfg, name=_nm,
                        subfolder=_save_subfolder, filename=_fname, debug=args.debug,
                    )
                    # debug=True so the destination path is always echoed — this is a
                    # published data product, not a debug aid.
                    save_df(
                        _merged, _dest, config=_cfg, name=_nm,
                        subfolder=_save_subfolder, filename=_fname,
                        rm=True, debug=args.debug,
                    )
        if args.debug:
            rprint(f"{_fname}.pkl written ({len(_contours_df)} grids)")
else:
    if args.debug:
        rprint("No contour grids collected — run 06_significance.py first.")
