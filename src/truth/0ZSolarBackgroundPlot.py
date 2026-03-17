import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

figure_path = f"{root}/images/background"
data_path = f"{root}/data/background"

for save_path in [figure_path, data_path]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of background particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

# Pickle load the data from the data/background/config folder
args = parser.parse_args()
user_input = {
    "rewrite": args.rewrite,
    "debug": args.debug,
}

rprint(f"\nRunning script: {os.path.abspath(__file__)} with config: {args.config}")

df = pickle.load(open(f"{root}/data/background/{args.config}/{args.config}_Background.pkl", "rb"))

fig = make_subplots(rows=1, cols=1)
id_df = df[(df["Plot"] == "ParticleID")]

for particle in id_df["Particle"].unique():
    particle_df = id_df[id_df["Particle"] == particle].explode(
        ["Counts", "Energy"]
    )
    fig.add_traces(
        go.Scatter(
            x=particle_df["Energy"],
            y=particle_df["Counts"],
            mode="lines",
            name=particle,
            line=dict(color=particle_df["Color"].iloc[0], width=2),
            legendgroup=particle_df["LegendGroup"].iloc[0],
            legendgrouptitle=dict(text=particle_df["LegendGroup"].iloc[0]),
            line_shape="hvh",
            line_dash=(
                "dash"
                if particle_df["LegendGroup"].iloc[0] != "DUNE Bkg.v3"
                else "solid"
            ),
        )
    )

fig = format_coustom_plotly(
    fig,
    title=f"Particle K.E. Distribution - {args.config}",
    log=(False, True),
    tickformat=(".1f", ".0e"),
    add_units=True,
    ranges=(None, [-1, 16]),
    legend=dict(x=0.78, y=0.99, font=dict(size=16)),
)
fig.update_xaxes(title_text="True Kinetic Energy (MeV)")
fig.update_yaxes(title_text="Counts (kT·year)⁻¹")

save_figure(
    fig,
    figure_path,
    args.config,
    None,
    filename=f"ParticleID_KE_Distribution",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

fig = make_subplots(rows=1, cols=1)
pdg_df = df[df["Plot"] == "ParticlePDG"]

for pdg_label in pdg_df["PDG"].unique():
    particle_df = pdg_df[pdg_df["PDG"] == pdg_label].explode(["Counts", "Energy"])
    fig.add_traces(
        go.Scatter(
            x=particle_df["Energy"],
            y=particle_df["Counts"],
            mode="lines",
            name=f"{pdg_label}",
            line=dict(color=particle_df["Color"].iloc[0], width=2),
            legendgroup=particle_df["LegendGroup"].iloc[0],
            legendgrouptitle=dict(text=particle_df["LegendGroup"].iloc[0]),
            line_shape="hvh",
            line_dash=(
                "dash"
                if particle_df["LegendGroup"].iloc[0] != "DUNE Bkg.v3"
                else "solid"
            ),
        )
    )

fig.update_traces(line=dict(width=2))
fig = format_coustom_plotly(
    fig,
    title=f"Particle K.E. Distribution - {args.config}",
    log=(False, True),
    tickformat=(".1f", ".0e"),
    add_units=True,
    ranges=(None, [-1, 16]),
    legend=dict(x=0.78, y=0.99, font=dict(size=16)),
)

fig.update_yaxes(title_text="Counts per Energy (kT·years·MeV)⁻¹")
fig.update_xaxes(title_text="True Particle K.E. (MeV)")
save_figure(
    fig,
    figure_path,
    args.config,
    None,
    filename=f"ParticlePDG_KE_Distribution",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)