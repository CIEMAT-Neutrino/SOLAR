def compute_particle_directions(run: dict, configs: dict, params: Optional[dict] = None, trees: list[str] = ["Reco"], rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    required_branches = {"Truth": ["Event", "Geometry", "Version"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "MTrackEnd", "MTrackStart", "MTrackDirection"]}
    new_branches = ["MTrackTheta", "MTrackPhi", "MTrackDirectionX",
                    "MTrackDirectionY", "MTrackDirectionZ", "MTrackDirectionMod"]
    for tree in trees:
        run[tree]["MTrackDirection"] = np.zeros(
            (len(run[tree]["Event"]), 3), dtype=np.float32)
        for branch in new_branches:
            run[tree][branch] = np.zeros(
                len(run[tree]["Event"]), dtype=np.float32)
        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            run[tree]["MTrackDirection"][idx] = np.subtract(
                run[tree]["MTrackEnd"][idx], run[tree]["MTrackStart"][idx])

            for coord_idx, coord in enumerate(["MTrackDirectionX", "MTrackDirectionY", "MTrackDirectionZ"]):
                if debug:
                    output += f"Computing {coord} direction"
                run[tree][coord][idx] = run[tree]["MTrackDirection"][:, coord_idx][idx]

            run[tree]["MTrackTheta"][idx] = np.arccos(
                run[tree]["MTrackDirectionZ"][idx])
            run[tree]["MTrackPhi"][idx] = np.arctan2(
                run[tree]["MTrackDirectionY"][idx], run[tree]["MTrackDirectionX"][idx])

        run = remove_branches(
            run, rm_branches, ["MTrackDirectionMod"], tree=tree, debug=debug)

    output += f"\tParticle direction computation \t-> Done!\n"
    return run, output, new_branches