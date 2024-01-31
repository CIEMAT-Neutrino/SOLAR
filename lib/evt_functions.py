import json

import numpy as np
import plotly.graph_objects as go
from particle import Particle
from plotly.subplots import make_subplots

from lib.plt_functions import format_coustom_plotly
from lib.geo_functions import add_geometry_planes
from lib.io_functions import get_bkg_config
from lib.solar_functions import get_pdg_color


def get_neutrino_positions(run, jdx):
    neut_x = [run["Reco"]["TNuX"][jdx]]
    neut_y = [run["Reco"]["TNuY"][jdx]]
    neut_z = [run["Reco"]["TNuZ"][jdx]]
    return neut_x, neut_y, neut_z


def get_marley_positions(run, jdx):
    ccint_x = [x for x in run["Reco"]["TMarleyX"][jdx] if x != 0]
    ccint_y = [x for x in run["Reco"]["TMarleyY"][jdx] if x != 0]
    ccint_z = [x for x in run["Reco"]["TMarleyZ"][jdx] if x != 0]
    true_name = [
        Particle.from_pdgid(x).name for x in run["Reco"]["TMarleyPDG"][jdx] if x != 0
    ]
    true_color = [
        get_pdg_color(int(x)) for x in run["Reco"]["TMarleyPDG"][jdx] if x != 0
    ]
    return ccint_x, ccint_y, ccint_z, true_name, true_color


def get_reco_positions(run, jdx, reco, color_dict):
    main_x = [run["Reco"][f"{reco}X"][jdx]]
    main_y = [run["Reco"][f"{reco}Y"][jdx]]
    main_z = [run["Reco"][f"{reco}Z"][jdx]]
    main_pdg = [Particle.from_pdgid(run["Reco"]["MainPDG"][jdx]).name]
    adj_x = [x for x in run["Reco"][f"AdjCl{reco}X"][jdx] if x != 0 and x > -1e6]
    adj_y = [x for x in run["Reco"][f"AdjCl{reco}Y"][jdx] if x != 0 and x > -1e6]
    adj_z = [x for x in run["Reco"][f"AdjCl{reco}Z"][jdx] if x != 0 and x > -1e6]
    reco_gen = [x for x in run["Reco"]["AdjClGen"][jdx] if x != 0]
    reco_name = [
        Particle.from_pdgid(x).name for x in run["Reco"]["AdjClMainPDG"][jdx] if x != 0
    ]
    reco_color = [
        get_pdg_color(int(run["Reco"]["AdjClMainPDG"][jdx][x]))
        if reco_gen[x] == 1
        else color_dict[reco_gen[x]]
        for x in range(len(reco_gen))
    ]
    return (
        main_x,
        main_y,
        main_z,
        main_pdg,
        adj_x,
        adj_y,
        adj_z,
        reco_gen,
        reco_name,
        reco_color,
    )


def plot_event(run, configs, jdx=None, tracked="Reco", debug=False):
    specs = [
        [
            {"type": "scatter"},
            {"type": "scatter"},
            {"type": "scatter3d", "colspan": 2},
            None,
        ]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs, subplot_titles=[""])

    for idx, config in enumerate(configs):
        info = json.load(open(f"../config/{config}/{config}_config.json"))
        bkg_dict, color_dict = get_bkg_config(info)
        if jdx is None:
            jdx = np.random.randint(len(run["Reco"]["Event"]))
            while (
                run["Reco"]["TNuE"][jdx] > 20
                or run["Reco"]["Primary"][jdx] != True
                or run["Reco"]["Generator"][jdx] != 1
            ):
                jdx = np.random.randint(len(run["Reco"]["Event"]))

        neut_x, neut_y, neut_z = get_neutrino_positions(run, jdx)
        ccint_x, ccint_y, ccint_z, true_name, true_color = get_marley_positions(
            run, jdx
        )
        (
            main_x,
            main_y,
            main_z,
            main_pdg,
            adj_x,
            adj_y,
            adj_z,
            reco_gen,
            reco_name,
            reco_color,
        ) = get_reco_positions(run, jdx, tracked, color_dict)

        fig.add_trace(
            go.Scatter3d(
                text="Neutrino",
                legendgrouptitle_text="Truth",
                legendgroup="0",
                marker_symbol="circle-open",
                x=neut_x,
                y=neut_y,
                z=neut_z,
                mode="markers",
                marker=dict(size=15, color="red"),
                name="Neutrino",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                text="Neutrino",
                legendgroup="0",
                marker_symbol="circle-open",
                x=neut_x,
                y=neut_y,
                mode="markers",
                marker=dict(size=20, color="red"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                text="Neutrino",
                legendgroup="0",
                marker_symbol="circle-open",
                x=neut_z,
                y=neut_y,
                mode="markers",
                marker=dict(size=20, color="red"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                text=main_pdg,
                legendgrouptitle_text="Reco",
                legendgroup="1",
                marker_symbol="circle",
                x=main_x,
                y=main_y,
                z=main_z,
                mode="markers",
                marker=dict(color="red"),
                name="Main",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                text=main_pdg,
                legendgroup="1",
                x=main_x,
                y=main_y,
                mode="markers",
                marker=dict(color="red"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                text=main_pdg,
                legendgroup="1",
                x=main_z,
                y=main_y,
                mode="markers",
                marker=dict(color="red"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                text=true_name,
                legendgroup="0",
                marker_symbol="square-open",
                x=ccint_x,
                y=ccint_y,
                z=ccint_z,
                mode="markers",
                marker=dict(color=true_color, size=10),
                name="Gamma",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                text=true_name,
                legendgroup="0",
                marker_symbol="square-open",
                x=ccint_x,
                y=ccint_y,
                mode="markers",
                marker=dict(color=true_color, size=15),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                text=true_name,
                legendgroup="0",
                marker_symbol="square-open",
                x=ccint_z,
                y=ccint_y,
                mode="markers",
                marker=dict(color=true_color, size=15),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                text=reco_name,
                legendgroup="1",
                marker_symbol="square",
                x=adj_x,
                y=adj_y,
                z=adj_z,
                mode="markers",
                marker=dict(color=reco_color),
                name="Adjacent",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                text=reco_name,
                legendgroup="1",
                marker_symbol="square",
                x=adj_x,
                y=adj_y,
                mode="markers",
                marker=dict(color=reco_color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                text=reco_name,
                legendgroup="1",
                marker_symbol="square",
                x=adj_z,
                y=adj_y,
                mode="markers",
                marker=dict(color=reco_color),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig = format_coustom_plotly(fig, figsize=(None, 600))
        fig.update_layout(
            title_text="TNuE: <b>%.2fMeV</b> Cluster: %i"
            % (run["Reco"]["TNuE"][jdx], jdx),
            title_x=0.5,
        )
        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], row=1, col=3)

        fig.update_yaxes(row=1, col=2, range=[neut_y[0] - 100, neut_y[0] + 100])
        fig.update_xaxes(row=1, col=1, range=[neut_z[0] - 100, neut_z[0] + 100])
        fig.update_xaxes(row=1, col=2, range=[neut_x[0] - 100, neut_x[0] + 100])

        fig.update_traces(hovertemplate="X: %{x} <br>Y: %{y} <br>PDG: %{text}")
        return fig
