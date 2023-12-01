import sys, json

sys.path.insert(0, "../")
from lib import (
    initialize_macro,
    check_macro_config,
    get_root_info,
    root2npy,
    remove_processed_branches,
)

default_dict = {}
user_input = initialize_macro(
    "00Process",
    ["config_file", "root_file", "rewrite", "trim", "debug"],
    default_dict=default_dict,
    debug=True,
)
user_input = check_macro_config(user_input, debug=user_input["debug"])

# Process data: root -> numpy #
info = json.load(open("../config/" + user_input["config_file"] + ".json", "r"))
for name in user_input["root_file"]:
    root_info = get_root_info(
        info["NAME"] + name, info["PATH"], debug=user_input["debug"]
    )
    if user_input["rewrite"] == False:
        root_info = remove_processed_branches(root_info, debug=user_input["debug"])
    root2npy(root_info, trim=user_input["trim"], debug=user_input["debug"])
