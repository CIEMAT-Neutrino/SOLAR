import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/event/opflash"

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
    "--index",
    type=int,
    help="The index of the event to plot",
    default=None,
)
parser.add_argument(
    "--tree",
    type=str,
    help="The tree to load",
    default="Reco",
    choices=["Reco", "Truth"],
)
parser.add_argument(
    "--variable",
    type=str,
    help="The variables to load",
    default="AdjOpFlash",
    choices=["AdjOpFlash", "OpFlash"],
)
parser.add_argument(
    "--signal",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="Signal flag (default: None)",
    choices=[True, False, None],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
configs = {args.config: [args.name]}

user_input = {"workflow": "ADJFLASH"}

run, output = load_multi(
    configs,
    load_all=False,
    preset=user_input["workflow"],
    generator_swap=False,
    debug=False,
)
rprint(output)

fig = plot_adjflash_event(
    run,
    configs,
    idx=args.index,
    tree=args.tree,
    tracked=args.variable,
    adjopflashsignal=args.signal,
    # adjopflashnum=10,
    adjopflashsize=100,
    zoom=False,
    debug=args.debug,
)

# fig.show()
fig = format_coustom_plotly(fig, add_watermark=False, debug=args.debug)
if args.save:
    save_figure(
        fig,
        path=save_path,
        config=args.config,
        name=args.name,
        filename=f'{args.variable}_Signal_{"True" if args.signal else "False"}_{args.index}',
        rm=args.rewrite,
        debug=args.debug,
    )
