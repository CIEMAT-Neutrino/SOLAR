import sys

sys.path.insert(0, "../../")
from lib import *

# Create a flag for debug mode that can be defined by the user when running the script python3 0XProcessOscillationFiles.py -d or python3 0XProcessOscillationFiles.py --debug
# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--debug", help="debug mode", action="store_true")
# args = parser.parse_args()
# debug = args.debug
path = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION"
debug = True

rprint(f"Searching for oscillation files in {path}/root/")
root_dm2, root_sin13, root_sin12 = get_oscillation_datafiles(
    dm2=None,
    sin13=None,
    sin12=None,
    path=f"{path}/root/",
    ext="root",
    auto=True,
    debug=debug,
)
rprint(f"Searching for oscillation files in {path}/pkl/rebin/")
pkl_dm2, pkl_sin13, pkl_sin12 = get_oscillation_datafiles(
    dm2=None,
    sin13=None,
    sin12=None,
    path=f"{path}/pkl/rebin/",
    ext="pkl",
    auto=True,
    debug=debug,
)

# for oscillation in zip(root_dm2, root_sin13, root_sin12):
for root_dm2, root_sin13, root_sin12 in track(
    zip(root_dm2, root_sin13, root_sin12),
    description="Processing oscillation files",
    total=len(root_dm2),
):
    oscillation = (root_dm2, root_sin13, root_sin12)
    # Check if this oscillation combination is exists in zip(pkl_dm2, pkl_sin13, pkl_sin12)
    if oscillation not in zip(pkl_dm2, pkl_sin13, pkl_sin12):
        print(f"Processing {oscillation}")
        dm2, sin13, sin12 = oscillation
        oscillation_df_dict = get_oscillation_map(
            dm2=dm2,
            sin13=sin13,
            sin12=sin12,
            auto=False,
            rebin=True,
            output="df",
            save=True,
            ext="root",
            debug=False,
        )
