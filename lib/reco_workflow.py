from typing import Optional
from rich import print as rprint

from .workflow.lib_main import compute_main_variables
from .workflow.lib_signal import (
    compute_signal_energies,
    compute_particle_energies,
    compute_signal_directions,
)
from .workflow.lib_efficiency import compute_true_efficiency, compute_particle_weights
from .workflow.lib_cluster import (
    compute_cluster_time,
    compute_electron_cluster,
    compute_cluster_calibration,
    compute_cluster_energy,
    compute_total_energy,
    compute_reco_energy,
    compute_energy_calibration,
    compute_cluster_adjrecox,
)
from .workflow.lib_adjcluster import compute_adjcl_basics, compute_adjcl_advanced
from .workflow.lib_adjopflash import compute_adjopflash_basics
from .workflow.lib_ophit import compute_ophit_basic, compute_ophit_event
from .workflow.lib_opflash import (
    compute_opflash_basic,
    compute_opflash_event,
    compute_opflash_main,
    compute_opflash_advanced,
    compute_opflash_matching,
)
from .workflow.lib_drift import compute_true_drift


def compute_reco_workflow(
    run: dict[dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    workflow: Optional[str] = None,
    rm_branches: bool = False,
    debug: bool = False,
) -> dict[dict]:
    """
    Compute the reco variables for the events in the run.
    All functions are called in the order they are defined in this file.
    All functions get the same arguments.

    Args:
        run: dictionary containing the run data.
        configs: dictionary containing the path to the configuration files for each geoemtry.
        params: dictionary containing the parameters for the reco functions.
        workflow: string containing the reco workflow to be used.
        rm_branches: boolean to remove the branches used in the reco workflow.
        debug: print debug information.

    Returns:
        run (dict): dictionary containing the TTree with the new branches.
    """
    # Compute reco variables
    new_branches = []
    output = f"[green]\nComputing {workflow} workflow:[/green]\n"

    if workflow is None:
        rprint(f"No workflow selected. Returning same run.\n")
        return run

    ### SIGNAL WORKFLOWS
    elif "TRUTH" in workflow:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
    elif "MARLEY" in workflow:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_signal_energies(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_signal_directions(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
    elif "RAW" in workflow:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_signal_energies(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_particle_energies(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches

    ### TPC WORKFLOWS
    elif "VERTEXING" in workflow:
        default_workflow_params = {
            "MAX_FLASH_R": None,
            "MIN_FLASH_PE": None,
            "RATIO_FLASH_PEvsR": None,
        }
        for key in default_workflow_params:
            if params is None:
                params = {}
            if key not in params:
                params[key] = default_workflow_params[key]

        run, output, this_new_branches = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_cluster_adjrecox(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
    elif "TRACK" in workflow:
        run, output, this_new_branches = compute_signal_directions(
            run,
            configs,
            params,
            trees=["Reco"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
    elif "ADJCL" in workflow:
        run, output, this_new_branches = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_cluster_energy(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_adjcl_advanced(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches

    ### PDS WORKFLOWS
    elif "OPHIT" in workflow:
        run, output, this_new_branches = compute_ophit_basic(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        run, output, this_new_branches = compute_ophit_event(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
    elif "OPFLASH" in workflow:
        run, output, this_new_branches = compute_opflash_basic(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_event(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_main(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
    elif "ADJFLASH" in workflow:
        run, output, this_new_branches = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_basic(
            run,
            configs,
            params,
            trees=["Reco"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_main(
            run,
            configs,
            params,
            trees=["Reco"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_adjopflash_basics(
            run,
            configs,
            params,
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
    elif "MATCHEDFLASH" in workflow:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches

    ### CALIBRATION AND RECONSTRUCTION WORKFLOWS
    elif [
        a in workflow
        for a in [
            "CORRECTION",
            "CALIBRATION",
            "DISCRIMINATION",
            "RECONSTRUCTION",
            "SMEARING",
            "ANALYSIS",
            "TEST",
        ]
    ].count(True) > 0:
        trees = ["Reco"]
        clusters = [""]
        if [a in workflow for a in ["SMEARING", "ANALYSIS"]].count(True) > 0:
            run, output, this_new_branches = compute_true_efficiency(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_main_variables(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        if [
            a in workflow for a in ["CORRECTION", "CALIBRATION", "DISCRIMINATION"]
        ].count(True) > 0:
            trees = ["Truth", "Reco"]
            run, output, this_new_branches = compute_adjcl_basics(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_electron_cluster(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["CORRECTION", "CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if [
            a in workflow
            for a in [
                "CORRECTION",
                "CALIBRATION",
                "DISCRIMINATION",
                "RECONSTRUCTION",
                "SMEARING",
            ]
        ].count(True) > 0:
            run, output, this_new_branches = compute_particle_energies(
                run,
                configs,
                params,
                trees=trees,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if [
            a in workflow
            for a in [
                "CALIBRATION",
                "DISCRIMINATION",
                "RECONSTRUCTION",
                "SMEARING",
                "ANALYSIS",
                "TEST",
            ]
        ].count(True) > 0:
            # if workflow == "CORRECTION" or workflow == "CALIBRATION":
            if (
                "CORRECTION" in workflow
                or "CALIBRATION" in workflow
                or "DISCRIMINATION" in workflow
            ):
                clusters.append("Electron")
            run, output, this_new_branches = compute_true_drift(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_cluster_time(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_cluster_energy(
                run,
                configs,
                params,
                clusters,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if [
            a in workflow
            for a in [
                "DISCRIMINATION",
                "RECONSTRUCTION",
                "SMEARING",
                "ANALYSIS",
                "TEST",
            ]
        ].count(True) > 0:
            run, output, this_new_branches = compute_cluster_calibration(
                run,
                configs,
                params,
                clusters,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_adjcl_basics(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_adjcl_advanced(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
            run, output, this_new_branches = compute_total_energy(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if [a in workflow for a in ["RECONSTRUCTION", "SMEARING", "ANALYSIS"]].count(
            True
        ) > 0:
            run, output, this_new_branches = compute_reco_energy(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["SMEARING", "ANALYSIS"]:
        if [a in workflow for a in ["SMEARING", "ANALYSIS"]].count(True) > 0:
            run, output, this_new_branches = compute_energy_calibration(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches

    rprint(output + f"[green]{workflow} workflow completed!\n[/green]")
    if debug:
        rprint(f"New branches: {new_branches}")
    return run
