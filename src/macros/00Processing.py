import sys, json

sys.path.insert(0, "../../")
from lib import (
    initialize_macro,
    check_macro_config,
    get_root_info,
    root2npy,
    remove_processed_branches,
    get_project_root
)

default_dict = {"rewrite": True, "trim": False, "debug": True}
root = get_project_root()
user_input = initialize_macro(
    "00Process",
    ["config_file", "root_prefix", "root_file", "rewrite", "trim", "debug"],
    default_dict=default_dict,
    debug=True,
)
user_input = check_macro_config(user_input, debug=user_input["debug"])

# Process data: root -> numpy #
info = json.load(open(f'{root}/config/{user_input["config_file"]}.json', "r"))
root_info = get_root_info(
    name=f'{user_input["root_prefix"]}{user_input["root_file"]}', path=f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/', user_input=user_input, debug=user_input["debug"]
)
if user_input["rewrite"] == False:
    root_info = remove_processed_branches(root_info, debug=user_input["debug"])
root2npy(root_info, user_input, trim=user_input["trim"], debug=user_input["debug"])
