from src.utils import get_project_root

import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from typing import Optional, Union
from particle import Particle
from plotly.subplots import make_subplots

from lib.plt_functions import format_coustom_plotly
from lib.geo_functions import add_geometry_planes
from lib.io_functions import get_bkg_config
from lib.solar_functions import get_pdg_color

compare = px.colors.qualitative.Plotly
root = get_project_root()


def get_flash_time(run, tree, idx, filter, debug=False):
    ophitPEs = [run[tree]["OpHitPE"][idx][x] for x in filter]
    ophit_times = [run[tree]["OpHitTime"][idx][x] for x in filter]
    flash_time = np.sum(np.multiply(ophit_times, ophitPEs)) / np.sum(ophitPEs)
    if debug:
        print(
            f"  Time: {2*flash_time:.2f} us ({np.min(ophit_times):.2f} : {np.max(ophit_times):.2f}) tick"
        )
    return flash_time


def get_ophit_positions(run, tree, idx, filter=None, debug=False):
    ophits = [[], [], []]
    # If filter is not None, then we filter the ophits by the filter array
    if filter is not None:
        flash_time = get_flash_time(run, tree, idx, filter, debug=debug)
        ophits[0] = [run[tree]["OpHitX"][idx][x] for x in filter]
        ophits[1] = [run[tree]["OpHitY"][idx][x] for x in filter]
        ophits[2] = [run[tree]["OpHitZ"][idx][x] for x in filter]
        return ophits
    # If filter is None, then we take all the ophits
    ophits[0] = run[tree]["OpHitX"][idx]
    ophits[1] = run[tree]["OpHitY"][idx]
    ophits[2] = run[tree]["OpHitZ"][idx]
    return ophits


def get_neutrino_positions(run, tree, idx):
    neut = [[], [], []]
    neut[0] = [run[tree]["SignalParticleX"][idx]]
    neut[1] = [run[tree]["SignalParticleY"][idx]]
    neut[2] = [run[tree]["SignalParticleZ"][idx]]
    neut_name = [Particle.from_pdgid(12).name]
    neut_color_dict = get_pdg_color(["12"])
    neut_color = list(neut_color_dict.values())[0]
    return neut, neut_name, neut_color


def get_signal_positions(run, tree, idx):
    ccint = [[], [], []]
    ccint[0] = [x for x in run[tree]["TSignalX"][idx] if x != 0]
    ccint[1] = [x for x in run[tree]["TSignalY"][idx] if x != 0]
    ccint[2] = [x for x in run[tree]["TSignalZ"][idx] if x != 0]
    true_name = [
        Particle.from_pdgid(x).name
        for x in run[tree]["TSignalPDG"][idx]
        if x not in [0, 1000190419]
    ]
    true_color = [
        list(get_pdg_color([x]).values())[0]
        for x in run[tree]["TSignalPDG"][idx]
        if x != 0
    ]
    return ccint, true_name, true_color


def get_edep_positions(run, tree, idx, max_edep_size=1000):
    edep = [[], [], []]
    edep[0] = [x for x in run[tree]["TSignalXDepList"][idx] if x != 0]
    edep[1] = [x for x in run[tree]["TSignalYDepList"][idx] if x != 0]
    edep[2] = [x for x in run[tree]["TSignalZDepList"][idx] if x != 0]
    edep_size = [x for x in run[tree]["TSignalEDepList"][idx] if x != 0]
    edep_size = [x if x < max_edep_size else max_edep_size for x in edep_size]
    edep_name = [
        Particle.from_pdgid(x).name
        for x in run[tree]["TSignalPDGDepList"][idx]
        if x not in [0, 1000190419]
    ]
    edep_color = [
        list(get_pdg_color([x]).values())[0]
        for x in run[tree]["TSignalPDGDepList"][idx]
        if x != 0
    ]
    return edep, edep_name, edep_color, edep_size


def get_main_positions(run, tree, idx, reco):
    main_vertex = [[], [], []]
    main_vertex[0] = [run[tree][f"{reco}X"][idx]]
    main_vertex[1] = [run[tree][f"{reco}Y"][idx]]
    main_vertex[2] = [run[tree][f"{reco}Z"][idx]]
    main_name = [Particle.from_pdgid(run[tree]["MainPDG"][idx]).name]
    main_color_dict = get_pdg_color([str(run[tree]["MainPDG"][idx])])
    main_color = list(main_color_dict.values())[0]
    return main_vertex, main_name, main_color


def get_adjflash_positions(run, tree, idx, flash, adjopflashsignal):
    if tree == "Truth":
        label = ""
    elif tree == "Reco":
        label = "Reco"

    main_vertex = [[], [], []]
    if adjopflashsignal is not None:
        if adjopflashsignal:
            jdx = np.where(
                (run[tree][f"{flash}Pur"][idx] > 0) * (run[tree][f"{flash}PE"][idx] > 0)
            )
        else:
            jdx = np.where(
                (run[tree][f"{flash}Pur"][idx] == 0)
                * (run[tree][f"{flash}PE"][idx] > 0)
            )
    else:
        jdx = np.where(
            (run[tree][f"{flash}Pur"][idx] != np.nan)
            * (run[tree][f"{flash}PE"][idx] > 0)
        )

    main_vertex[0] = [x for x in run[tree][f"{flash}{label}X"][idx][jdx] if x > -1e6]
    main_vertex[1] = [x for x in run[tree][f"{flash}{label}Y"][idx][jdx] if x > -1e6]
    main_vertex[2] = [x for x in run[tree][f"{flash}{label}Z"][idx][jdx] if x > -1e6]
    main_size = [x for x in run[tree][f"{flash}PE"][idx][jdx] if x > 0]
    main_name = [
        "Signal" if x == True else "Background"
        for x in run[tree][f"{flash}Pur"][idx][jdx] > 0
    ]
    main_color = run[tree][f"{flash}Pur"][idx][jdx]
    return main_vertex, main_name, main_color, main_size


def get_adjcl_positions(run, tree, idx, reco, color_dict, get_adjacent_color=False):
    adj_vertex = [[], [], []]
    adj_vertex[0] = [x for x in run[tree][f"AdjCl{reco}X"][idx] if x != 0 and x > -1e6]
    adj_vertex[1] = [x for x in run[tree][f"AdjCl{reco}Y"][idx] if x != 0 and x > -1e6]
    adj_vertex[2] = [x for x in run[tree][f"AdjCl{reco}Z"][idx] if x != 0 and x > -1e6]
    reco_gen = [x for x in run[tree]["AdjClGen"][idx] if x != 0]
    reco_name = [
        Particle.from_pdgid(x).name for x in run[tree]["AdjClMainPDG"][idx] if x != 0
    ]
    if get_adjacent_color:
        reco_color = [
            (
                list(get_pdg_color([str(run[tree]["AdjClMainPDG"][idx][x])]).values())[
                    0
                ]
                if reco_gen[x] == 1
                else color_dict[reco_gen[x]]
            )
            for x in range(len(reco_gen))
        ]
    else:
        # Color defined by the reco_gen red for 1 and blue for else
        reco_color = [
            (
                list(get_pdg_color([str(run[tree]["AdjClMainPDG"][idx][x])]).values())[
                    0
                ]
                if reco_gen[x] == 1
                else "black"
            )
            for x in range(len(reco_gen))
        ]
    return adj_vertex, reco_name, reco_color


def add_data_to_event(
    fig,
    geometry,
    data,
    idx,
    title,
    subtitle,
    name,
    symbol: Union[str, list] = "circle",
    size: Union[int, list] = 20,
    color: Union[str, list] = "black",
    options: dict = {"lw": 0, "marker": None, "colorscale": "Turbo"},
    debug: bool = False,
):
    default_options = {"lw": 0, "marker": None, "colorscale": "Turbo"}
    for key in options:
        try:
            default_options[key] = options[key]
        except KeyError:
            print(f"Key {key} not found in options")

    for coord in data:
        # Search fo rvalues in coord that are <= -1e6 and set them to 0
        coord[:] = [x if x > -1e6 else 0 for x in coord]

    if geometry == "vd":
        x, y, z = data[0], data[1], data[2]
    elif geometry == "hd":
        x, y, z = data[0], data[1], data[2]

    # print(data)
    fig.add_trace(
        go.Scatter3d(
            text=name,
            name=subtitle,
            legendgrouptitle_text=title,
            legendgroup=idx,
            marker_symbol=symbol,
            x=data[0],
            y=data[1],
            z=data[2],
            mode="markers",
            marker=dict(
                size=(
                    [min(i * 10, 10) for i in size]
                    if isinstance(size, (list, np.ndarray))
                    else min(size, 10)
                ),
                color=color,
                cmin=0,
                cmax=1,
                colorscale=default_options["colorscale"],
                line_width=default_options["lw"],
            ),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            text=name,
            legendgroup=idx,
            marker_symbol=symbol,
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=np.asarray(size) + 5,
                color=color,
                cmin=0,
                cmax=1,
                colorscale=default_options["colorscale"],
                line_width=default_options["lw"],
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            text=name,
            legendgroup=idx,
            marker_symbol=symbol,
            x=z,
            y=y,
            mode="markers",
            marker=dict(
                size=np.asarray(size) + 5,
                color=color,
                cmin=0,
                cmax=1,
                colorscale=default_options["colorscale"],
                line_width=default_options["lw"],
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    return fig


def add_particle_legend(fig, run, tracked, varaibales: dict, idx):
    # neutrino_color_dict = get_pdg_color(
    #     [str(run[tracked]["SignalParticlePDG"][idx])]
    # )
    # signal_color_dict = get_pdg_color(
    #     [str(x) for x in run[tracked]["TSignalPDG"][idx] if x != 0]
    # )
    # main_color_dict = get_pdg_color(
    #     [str(x) for x in run[tracked]["AdjClMainPDG"][idx] if x != 0]
    # )
    dict_list = []
    # Loop over entries in varaibales dict. Key is variable name, argument is the variable type
    for variable in varaibales:
        if varaibales[variable] == "array":
            this_dict = get_pdg_color([str(run[tracked][variable][idx])])
        elif varaibales[variable] == "list":
            this_dict = get_pdg_color(
                [str(x) for x in run[tracked][variable][idx] if x != 0]
            )
        dict_list.append(this_dict)
    dict_list[-1]["background"] = "black"

    # color_dict = {**neutrino_color_dict, **signal_color_dict, **main_color_dict}
    color_dict = {}
    for d in dict_list:
        color_dict = {**color_dict, **d}

    for key, value in color_dict.items():
        if key != "background":
            key = Particle.from_pdgid(key).name
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=value, symbol="circle"),
                name=key,
                legendgroup="Particle",
                legendgrouptitle_text="Particle Color",
            ),
            row=1,
            col=1,
        )
    return fig


def plot_edep_event(run, configs, idx=None, tracked="Truth", zoom=True, debug=False):
    specs = [
        [
            {"type": "scatter"},
            {"type": "scatter"},
            {"type": "scatter3d", "colspan": 2},
            None,
        ]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs, subplot_titles=[""])

    for i, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
        bkg_dict, color_dict = get_bkg_config(info)
        if idx is None:
            jdx = np.where((run[tracked]["SignalParticleK"] <= 20))
            idx = np.random.randint(len(run[tracked]["Event"][jdx]))
            idx = jdx[0][idx]
        else:
            if debug:
                print(f"**Event: {int(idx)}")
                print(
                    f"**Particle K.E.: {run[tracked]['SignalParticleK'][idx]:.2f} MeV"
                )

        # neut, neut_name, neut_color = get_neutrino_positions(run, tracked, idx)
        ccint, true_name, true_color = get_signal_positions(run, tracked, idx)
        edep, edep_name, edep_color, edep_size = get_edep_positions(run, tracked, idx)
        # adj, adjcl_name, adjcl_color = get_adjcl_positions(run, tracked, idx, tracked, color_dict)

        # fig = add_data_to_event(fig, neut, "0", "Truth", "Neutrino", neut_name, "circle-open", 15, neut_color, {"lw":1})
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            edep,
            "1",
            "Raw",
            "EDeps",
            edep_name,
            "circle",
            2000 * edep_size,
            edep_color,
            {},
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            ccint,
            "0",
            "Reco",
            "Particles",
            true_name,
            "square-open",
            20,
            true_color,
            {"lw": 2},
        )
        fig = add_particle_legend(
            fig, run, tracked, {"SignalParticlePDG": "array", "TSignalPDG": "list"}, idx
        )
        fig = format_coustom_plotly(
            fig, figsize=(None, 600), tickformat=(".1f", ".1f"), add_watermark=False
        )

        particle = Particle.from_pdgid(run[tracked]["TSignalPDG"][:, 0][idx]).name
        fig.update_layout(
            title_text="Neutrino <b>%.2f MeV</b> Cluster: %i"
            % (run[tracked]["SignalParticleE"][idx], idx),
            title_x=0.5,
        )

        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], row=1, col=3)

        if zoom:
            fig.update_xaxes(row=1, col=2, range=[ccint[0][0] - 100, ccint[0][0] + 100])
            fig.update_yaxes(row=1, col=2, range=[ccint[1][0] - 100, ccint[1][0] + 100])
            fig.update_xaxes(row=1, col=1, range=[ccint[2][0] - 100, ccint[2][0] + 100])

        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f} <br>PDG: %{text}")
        return fig, idx


def plot_adjflash_event(
    run,
    configs,
    idx=None,
    tree="Truth",
    tracked="AdjOpFlash",
    unzoom=True,
    adjopflashsignal: Optional[bool] = None,
    adjopflashsize: Optional[int] = None,
    debug: bool = False,
):
    specs = [
        [
            {"type": "scatter"},
            {"type": "scatter"},
            {"type": "scatter3d", "colspan": 2},
            None,
        ]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs, subplot_titles=[""])
    run[tree][f"{tracked}Num"] = np.sum(
        run[tree][f"{tracked}PE"] > 0, axis=1
    )  # Get the number of adjacent flashes
    for i, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
        bkg_dict, color_dict = get_bkg_config(info)
        if idx is None:
            idx = np.random.randint(len(run["Reco"]["Event"]))
            if tree == "Reco":
                while (
                    run["Reco"]["Primary"][idx] != True
                    or run["Reco"]["Generator"][idx] != 1
                    # or run["Reco"]["AdjOpFlashNum"][idx] <= adjopflashnum
                    # or abs(run["Reco"]["RecoX"][idx]) > info["DETECTOR_SIZE_X"] / 2
                ):
                    idx = np.random.randint(len(run["Reco"]["Event"]))

        neut, neut_name, neut_color = get_neutrino_positions(run, tree, idx)
        if tree == "Reco":
            main, main_name, main_color = get_main_positions(run, tree, idx, tree)
        flash, flash_name, flash_color, flash_size = get_adjflash_positions(
            run, tree, idx, tracked, adjopflashsignal
        )
        if adjopflashsize is not None:
            flash_size = [
                x if x < adjopflashsize else adjopflashsize for x in flash_size
            ]

        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            neut,
            "0",
            "Truth",
            "Neutrino",
            neut_name,
            "circle-open",
            15,
            neut_color,
            {"lw": 1},
        )
        if tree == "Reco":
            fig = add_data_to_event(
                fig,
                info["GEOMETRY"],
                main,
                "1",
                "Reco",
                "Cluster",
                main_name,
                "circle",
                10,
                main_color,
                {"lw": 1},
            )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            flash,
            "1",
            "Reco",
            "AdjOpFlash",
            flash_name,
            "x",
            0.1 * np.asarray(flash_size),
            flash_color,
            {"lw": 1},
        )

        fig = format_coustom_plotly(fig, figsize=(None, 600), add_watermark=False)
        fig.update_layout(
            title_text="Particle K.E.: <b>%.2f MeV</b> Cluster: %i %s"
            % (run[tree]["SignalParticleK"][idx], idx, config),
            title_x=0.5,
        )
        if info["GEOMETRY"] == "hd":
            fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
            fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
            fig.update_yaxes(title_text="Y [cm]", row=1, col=1)

        elif info["GEOMETRY"] == "vd":
            fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
            fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
            fig.update_yaxes(title_text="Y [cm]", row=1, col=1)

        fig = add_geometry_planes(fig, info["GEOMETRY"], unzoom=unzoom, row=1, col=3)

        # if zoom:
        #     fig.update_xaxes(row=1, col=2, range=[neut[0][0] - 100, neut[0][0] + 100])
        #     fig.update_yaxes(row=1, col=2, range=[neut[1][0] - 100, neut[1][0] + 100])
        #     fig.update_xaxes(row=1, col=1, range=[neut[2][0] - 100, neut[2][0] + 100])

        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f} <br>PDG: %{text}")
        return fig, idx


def plot_tpc_event(
    run,
    configs,
    idx=None,
    tracked="Reco",
    zoom=True,
    adjclnum=0,
    get_adj_color=False,
    unzoom=1,
    debug=False,
):
    specs = [
        [
            {"type": "scatter"},
            {"type": "scatter"},
            {"type": "scatter3d", "colspan": 2},
            None,
        ]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs, subplot_titles=[""])

    for i, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
        bkg_dict, color_dict = get_bkg_config(info)
        if idx is None:
            jdx = np.where((run["Reco"]["AdjClNum"] >= adjclnum))
            idx = np.random.randint(len(run["Reco"]["Event"][jdx]))
            idx = jdx[0][idx]

        neut, neut_name, neut_color = get_neutrino_positions(run, tracked, idx)
        ccint, true_name, true_color = get_signal_positions(run, tracked, idx)
        main, main_name, main_color = get_main_positions(run, tracked, idx, tracked)
        adj, adjcl_name, adjcl_color = get_adjcl_positions(
            run, tracked, idx, tracked, color_dict, get_adjacent_color=False
        )
        if debug:
            print(f"**True X: {neut[0][0]:.2f} Y: {neut[1][0]:.2f} Z: {neut[2][0]:.2f}")
            print(
                f"**Reco X: {ccint[0][0]:.2f} Y: {ccint[1][0]:.2f} Z: {ccint[2][0]:.2f}"
            )

        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            neut,
            "0",
            "Signal Truth",
            "neutrino",
            neut_name,
            "circle-open",
            15,
            neut_color,
            {"lw": 1},
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            ccint,
            "0",
            "Signal Truth",
            "daughter",
            true_name,
            "square-open",
            15,
            true_color,
            {"lw": 1},
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            main,
            "1",
            "Reco Cluster",
            "main",
            main_name,
            "circle",
            10,
            main_color,
            {"lw": 1},
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            adj,
            "1",
            "Reco Cluster",
            "adjacent",
            adjcl_name,
            "square",
            10,
            adjcl_color,
        )

        fig = format_coustom_plotly(fig, figsize=(None, 600), add_watermark=False)
        fig.update_layout(
            title_text="Particle K.E.: <b>%.2f MeV</b> Cluster: %i %s"
            % (run["Reco"]["SignalParticleK"][idx], idx, config),
            title_x=0.5,
        )
        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], unzoom, row=1, col=3)

        if zoom:
            fig.update_xaxes(row=1, col=2, range=[neut[0][0] - 100, neut[0][0] + 100])
            fig.update_yaxes(row=1, col=2, range=[neut[1][0] - 100, neut[1][0] + 100])
            fig.update_xaxes(row=1, col=1, range=[neut[2][0] - 100, neut[2][0] + 100])

        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f} <br>PDG: %{text}")

        fig = add_particle_legend(
            fig,
            run,
            tracked,
            {
                "SignalParticlePDG": "array",
                "TSignalPDG": "list",
                "AdjClMainPDG": "list",
            },
            idx,
        )

        return fig, idx


def plot_pds_event(
    run,
    configs,
    idx=None,
    tracked="Truth",
    maxophit=100,
    flashid: int = None,
    zoom=True,
    debug=False,
):
    colorscale = [[0, "green"], [0.5, "yellow"], [1, "orange"]]
    specs = [
        [
            {"type": "scatter"},
            {"type": "scatter"},
            {"type": "scatter3d", "colspan": 2},
            None,
        ]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs, subplot_titles=[""])

    run[tracked]["OpHitFlashPur"] = np.copy(run[tracked]["OpHitPur"])
    for i, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
        if idx is None:
            idx = np.random.randint(len(run[tracked]["Event"]))
            while run[tracked]["SignalParticleE"][idx] > 30:
                idx = np.random.randint(len(run[tracked]["Event"]))

        neut, neut_name, neut_color = get_neutrino_positions(run, tracked, idx)
        ccint, true_name, true_color = get_signal_positions(run, tracked, idx)
        # If flashid is not None, then we filter the ophits by the flashid
        if flashid == "All":
            ophits = get_ophit_positions(run, tracked, idx)
            ophit_size = [int(x) if x > 0 else 0 for x in run[tracked]["OpHitPE"][idx]]
            ophit_size = [x if x < maxophit else maxophit for x in ophit_size]
            ophit_color = [
                float(x) if x > 0 else 0 for x in run[tracked]["OpHitPur"][idx]
            ]
        else:
            # Set all the flash purities to 0
            run[tracked]["OpHitFlashPur"][idx] = np.zeros(
                len(run[tracked]["OpHitPur"][idx])
            )
            for i in range(0, int(np.max(run[tracked]["OpHitFlashID"][idx]))):
                this_flash_filter = np.where(
                    np.array(run[tracked]["OpHitFlashID"][idx]) == i
                )[0]
                this_flash_pur = np.sum(
                    np.multiply(
                        run[tracked]["OpHitPur"][idx][this_flash_filter],
                        run[tracked]["OpHitPE"][idx][this_flash_filter],
                    )
                ) / np.sum(run[tracked]["OpHitPE"][idx][this_flash_filter])
                run[tracked]["OpHitFlashPur"][idx][this_flash_filter] = this_flash_pur

            if flashid is not None:
                flash_filter = np.where(
                    np.array(run[tracked]["OpHitFlashID"][idx]) == flashid
                )[0]
            if flashid is None:
                flash_max_pur = np.max(
                    run[tracked]["OpHitFlashPur"][idx],
                    initial=0,
                    where=run[tracked]["OpHitFlashPur"][idx] > 0,
                )
                flash_filter = np.where(
                    np.array(run[tracked]["OpHitFlashPur"][idx]) == flash_max_pur
                )[0]
                flashid = run[tracked]["OpHitFlashID"][idx][flash_filter][0]

            ophit_size = [
                int(x) if x > 0 else 0
                for x in run[tracked]["OpHitPE"][idx][flash_filter]
            ]
            ophit_size = [x if x < maxophit else maxophit for x in ophit_size]
            ophit_color = [
                float(x) if x > 0 else 0
                for x in run[tracked]["OpHitPur"][idx][flash_filter]
            ]

            if debug:
                print(f"**Event: {int(idx)}")
                print(f"**Flash ID: {int(flashid)}")
                print(f"  Size: {np.sum(ophit_size)} PE")
                print(
                    f"  Purity: {run[tracked]['OpHitFlashPur'][idx][flash_filter][0]:.2f}"
                )

            ophits = get_ophit_positions(run, tracked, idx, flash_filter, debug=debug)

            if debug:
                # print(f"Flash Purity: {np.sum(np.multiply(ophit_color,ophit_size))/np.sum(ophit_size):.2f}")
                print(
                    f"  Vertex (Y,Z): {np.sum(np.multiply(ophits[1],ophit_size))/np.sum(ophit_size):.2f}, {np.sum(np.multiply(ophits[2],ophit_size))/np.sum(ophit_size):.2f}"
                )

        flash = [[], [], []]
        flash[0] = [np.sum(np.multiply(ophits[0], ophit_size)) / np.sum(ophit_size)]
        flash[1] = [np.sum(np.multiply(ophits[1], ophit_size)) / np.sum(ophit_size)]
        flash[2] = [np.sum(np.multiply(ophits[2], ophit_size)) / np.sum(ophit_size)]
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            neut,
            "0",
            "Truth",
            "Neutrino",
            neut_name,
            "circle-open",
            20,
            neut_color,
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            ccint,
            "0",
            "Truth",
            "Daughter",
            true_name,
            "square-open",
            10,
            true_color,
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            ophits,
            "1",
            "Reco",
            "Ophits",
            "Ophits",
            "circle",
            ophit_size,
            ophit_color,
        )
        fig = add_data_to_event(
            fig,
            info["GEOMETRY"],
            flash,
            "1",
            "Reco",
            "Flash",
            "Flash",
            "circle",
            10,
            "red",
        )

        fig = format_coustom_plotly(fig, figsize=(None, 600), add_watermark=False)
        fig.update_layout(
            coloraxis=dict(colorscale=colorscale),
            title_text="SignalParticleE: <b>%.2f MeV</b> Event: %i FlashID: %i Purity: %.2f"
            % (
                run[tracked]["SignalParticleE"][idx],
                idx,
                flashid,
                np.sum(np.multiply(ophit_color, ophit_size)) / np.sum(ophit_size),
            ),
            title_x=0.5,
        )
        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], unzoom=2, row=1, col=3)
        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f}")

    return fig, idx
