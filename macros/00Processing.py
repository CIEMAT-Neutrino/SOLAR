import sys
sys.path.insert(0, '../')
from lib import initialize_macro, check_macro_config, read_input_file, get_root_info, root2npy, remove_processed_branches

default_dict = {}
user_input = initialize_macro("00Process",["config_file","root_file","rewrite","trim","debug"],default_dict=default_dict, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Process data: root -> numpy #
info = read_input_file(user_input["config_file"],path="../config/",debug=user_input["debug"])
for name in user_input["root_file"]:
    root_info = get_root_info(info["NAME"][0]+name,info["PATH"][0],debug=user_input["debug"])
    if user_input["rewrite"] == False: root_info = remove_processed_branches(root_info,debug=user_input["debug"])
    root2npy(root_info,trim=user_input["trim"],debug=user_input["debug"])