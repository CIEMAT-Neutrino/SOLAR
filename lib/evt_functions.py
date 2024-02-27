import json

import numpy as np
import plotly.graph_objects as go
from particle import Particle
from plotly.subplots import make_subplots

from lib.plt_functions import format_coustom_plotly
from lib.geo_functions import add_geometry_planes
from lib.io_functions import get_bkg_config
from lib.solar_functions import get_pdg_color

def get_ophit_positions(run, tree, idx):
    ophits = [[],[],[]]
    ophits[0] = run[tree]["OpHitX"][idx]
    ophits[1] = run[tree]["OpHitY"][idx]
    ophits[2] = run[tree]["OpHitZ"][idx]
    return ophits


def get_neutrino_positions(run, tree, idx):
    neut = [[],[],[]]
    neut[0] = [run[tree]["TNuX"][idx]]
    neut[1] = [run[tree]["TNuY"][idx]]
    neut[2] = [run[tree]["TNuZ"][idx]]
    neut_name = [Particle.from_pdgid(12).name]
    neut_color = get_pdg_color(12)
    return neut, neut_name, neut_color


def get_marley_positions(run, tree, idx):
    ccint = [[],[],[]]
    ccint[0] = [x for x in run[tree]["TMarleyX"][idx] if x != 0]
    ccint[1] = [x for x in run[tree]["TMarleyY"][idx] if x != 0]
    ccint[2] = [x for x in run[tree]["TMarleyZ"][idx] if x != 0]
    true_name = [
        Particle.from_pdgid(x).name for x in run[tree]["TMarleyPDG"][idx] if x != 0
    ]
    true_color = [
        get_pdg_color(int(x)) for x in run[tree]["TMarleyPDG"][idx] if x != 0
    ]
    return ccint, true_name, true_color


def get_main_positions(run, tree, idx, reco, color_dict):
    main_vertex = [[],[],[]]
    main_vertex[0] = [run[tree][f"{reco}X"][idx]]
    main_vertex[1] = [run[tree][f"{reco}Y"][idx]]
    main_vertex[2] = [run[tree][f"{reco}Z"][idx]]
    main_name = [Particle.from_pdgid(run[tree]["MainPDG"][idx]).name]
    main_color = get_pdg_color(int(run[tree]["MainPDG"][idx]))
    return main_vertex,main_name,main_color


def get_adjcl_positions(run, tree, idx, reco, color_dict):
    adj_vertex = [[],[],[]]
    adj_vertex[0] = [x for x in run[tree][f"AdjCl{reco}X"][idx] if x != 0 and x > -1e6]
    adj_vertex[1] = [x for x in run[tree][f"AdjCl{reco}Y"][idx] if x != 0 and x > -1e6]
    adj_vertex[2] = [x for x in run[tree][f"AdjCl{reco}Z"][idx] if x != 0 and x > -1e6]
    reco_gen = [x for x in run[tree]["AdjClGen"][idx] if x != 0]
    reco_name = [
        Particle.from_pdgid(x).name for x in run[tree]["AdjClMainPDG"][idx] if x != 0
    ]
    reco_color = [
        get_pdg_color(int(run[tree]["AdjClMainPDG"][idx][x]))
        if reco_gen[x] == 1
        else color_dict[reco_gen[x]]
        for x in range(len(reco_gen))
    ]
    return adj_vertex,reco_name,reco_color


def add_data_to_event(fig, data, idx, title, subtitle, name, symbol, size, color, options:dict = {}, debug:bool = False):
    default_options = {"lw":0, "marker":None}
    for key in options:
        try:
            default_options[key] = options[key]
        except KeyError:
            print(f"Key {key} not found in options")
        
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
            marker=dict(size=np.asarray(size), 
                color=color, 
                colorscale="Turbo",
                line_width=default_options["lw"]
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
            x=data[0],
            y=data[1],
            mode="markers",
            marker=dict(size=np.asarray(size)+5,
                color=color,
                colorscale="Turbo",
                line_width=default_options["lw"]
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
            x=data[2],
            y=data[1],
            mode="markers",
            marker=dict(size=np.asarray(size)+5,
                color=color,
                colorscale="Turbo",
                line_width=default_options["lw"]),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    return fig

def plot_tpc_event(run, configs, idx=None, tracked="Reco", zoom = True, debug=False):
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
        info = json.load(open(f"../config/{config}/{config}_config.json"))
        bkg_dict, color_dict = get_bkg_config(info)
        if idx is None:
            idx = np.random.randint(len(run["Reco"]["Event"]))
            while (
                run["Reco"]["TNuE"][idx] > 30
                or run["Reco"]["Primary"][idx] != True
                or run["Reco"]["Generator"][idx] != 1
            ):
                idx = np.random.randint(len(run["Reco"]["Event"]))

        neut, neut_name, neut_color = get_neutrino_positions(run, tracked, idx)
        ccint, true_name, true_color = get_marley_positions(run, tracked, idx)
        main, main_name, main_color = get_main_positions(run, tracked, idx, tracked, color_dict)
        adj, adjcl_name, adjcl_color = get_adjcl_positions(run, tracked, idx, tracked, color_dict)

        fig = add_data_to_event(fig, neut, "0", "Truth", "Neutrino", neut_name, "circle-open", 15, neut_color, {"lw":1})
        fig = add_data_to_event(fig, ccint, "0", "Truth", "Daughter", true_name,"square-open", 15, true_color,{"lw":1})
        fig = add_data_to_event(fig, main, "1", "Reco", "Main", main_name,"circle", 10, main_color, {"lw":1})
        fig = add_data_to_event(fig, adj, "1", "Reco", "Adjacent", adjcl_name,"square", 10, adjcl_color)

        fig = format_coustom_plotly(fig, figsize=(None, 600))
        fig.update_layout(
            title_text="TNuE: <b>%.2fMeV</b> Cluster: %i"
            % (run["Reco"]["TNuE"][idx], idx),
            title_x=0.5,
        )
        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], row=1, col=3)
        
        if zoom:
            fig.update_xaxes(row=1, col=2, range=[neut[0][0] - 100, neut[0][0] + 100])
            fig.update_yaxes(row=1, col=2, range=[neut[1][0] - 100, neut[1][0] + 100])
            fig.update_xaxes(row=1, col=1, range=[neut[2][0] - 100, neut[2][0] + 100])

        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f} <br>PDG: %{text}")
        return fig
    

def plot_pds_event(run, configs, idx=None, tracked="Truth", zoom = True, debug=False):
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
        info = json.load(open(f"../config/{config}/{config}_config.json"))
        if idx is None:
            idx = np.random.randint(len(run[tracked]["Event"]))
            while (run[tracked]["TNuE"][idx] > 30):
                idx = np.random.randint(len(run[tracked]["Event"]))

        neut, neut_name, neut_color = get_neutrino_positions(run, tracked, idx)
        ccint, true_name, true_color = get_marley_positions(run, tracked, idx)
        ophits = get_ophit_positions(run, tracked, idx)
        fig = add_data_to_event(fig, neut, "0", "Truth", "Neutrino", neut_name, "circle-open", 20, neut_color)
        fig = add_data_to_event(fig, ccint, "0", "Truth", "Daughter", true_name,"square-open", 20, true_color)
        # Make a ophit_size array from the ophitPE branch of run substituting the numbers < 0 with 0
        ophit_size = [int(x) if x > 0 else 0 for x in run[tracked]["OpHitPE"][idx]]
        ophit_color = [float(x) if x > 0 else 0 for x in run[tracked]["OpHitPur"][idx]]
        fig = add_data_to_event(fig, ophits, "1", "Reco", "Ophits", "Ophits","circle", ophit_size, ophit_color)

        fig = format_coustom_plotly(fig, figsize=(None, 600))
        fig.update_layout(
            title_text="TNuE: <b>%.2fMeV</b> Cluster: %i"
            % (run["Reco"]["TNuE"][idx], idx),
            title_x=0.5,
        )
        fig.update_xaxes(matches=None, title_text="Z [cm]", row=1, col=1)
        fig.update_xaxes(matches=None, title_text="X [cm]", row=1, col=2)
        fig.update_yaxes(title_text="Y [cm]", row=1, col=1)
        fig = add_geometry_planes(fig, info["GEOMETRY"], row=1, col=3)
        fig.update_traces(hovertemplate="X: %{x:.2f} <br>Y: %{y:.2f}")

    return fig