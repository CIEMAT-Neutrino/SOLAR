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
    "--name", type=str, help="The name of the configuration", default="marley_official"
)
parser.add_argument("--tree", type=str, default=["Config", "Truth", "Reco"], nargs="+")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "CORRECTION",
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs,
    tree_labels=args.tree,
    load_all=True,
    debug=user_input["debug"],
)

# Run is a dict of dicts. Inside each dict are branches that can be of any type.
# Print a summary of the loaded information in run to the terminal.
rprint(f"Loaded production: {config} - {name}\n")
for tree in args.tree:
    rprint(f"Tree: {tree}, Number of branches: {len(run[tree])}")
    df = pd.DataFrame()
    df_list = []
    for branch in run[tree]:
        branch_data = run[tree][branch]
        if isinstance(branch_data, np.ndarray):
            shape = branch_data.shape
            dtype = branch_data.dtype
            if isinstance(branch_data[0], str):
                value = branch_data[0]
            else:
                try:
                    value = branch_data[0][0]
                except IndexError:
                    value = branch_data[0]
        elif isinstance(branch_data, list):
            length = len(branch_data)
            try:
                value = branch_data[0][0]
            except IndexError:
                value = branch_data[0]
        else:
            value = branch_data
            rprint(f"\t{branch}: {type(branch_data).__name__}")
        df_list.append(
            pd.DataFrame.from_dict(
                {
                    "Branch": branch,
                    "Value": value,
                    "Type": type(branch_data).__name__,
                    "Shape": shape if isinstance(branch_data, np.ndarray) else None,
                    "Dtype": dtype if isinstance(branch_data, np.ndarray) else None,
                },
                orient="index",
            ).T
        )
    df = pd.concat(df_list, ignore_index=True)
    pd.set_option("display.max_rows", None)
    rprint(df)
    rprint("\n")
# End of file
