import sys

sys.path.insert(0, "../")

from lib import *

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Print the branches of a ROOT file",
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument("--tree", type=str, default=["Config", "Truth", "Reco"], nargs="+")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

config = parser.parse_args().config
name = parser.parse_args().name

configs = {config: [name]}

user_input = {
    "workflow": "CORRECTION",
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}

run, output = load_multi(
    configs,
    tree_labels=parser.parse_args().tree,
    load_all=True,
    debug=user_input["debug"],
)

# Run is a dict of dicts. Inside each dict are branches that can be of any type.
# Print a summary of the loaded information in run to the terminal.

for tree in run:
    rprint(f"Tree: {tree}, Number of branches: {len(run[tree])}")
    for branch in run[tree]:
        branch_data = run[tree][branch]
        if isinstance(branch_data, np.ndarray):
            shape = branch_data.shape
            dtype = branch_data.dtype
            rprint(f"\t{branch}: ndarray, Shape: {shape}, Dtype: {dtype}")
        elif isinstance(branch_data, list):
            length = len(branch_data)
            rprint(f"\t{branch}: list, Length: {length}")
        elif isinstance(branch_data, (int, float, str)):
            rprint(f"\t{branch}: {type(branch_data).__name__}, Value: {branch_data}")
        else:
            rprint(f"\t{branch}: {type(branch_data).__name__}")
    rprint("\n")
# End of file
