from typing import Optional, Union
from rich import print as rprint

from .main import (
    compute_main_variables,
    update_default_values,
    split_vector_branches,
)
from .signal import (
    compute_signal_energies,
    compute_particle_energies,
    compute_signal_directions,
)
from .efficiency import compute_true_efficiency
from .weights import compute_particle_weights, compute_particle_surface
from .cluster import (
    compute_cluster_time,
    compute_electron_cluster,
    compute_cluster_calibration,
    compute_cluster_energy,
    compute_total_energy,
    compute_reco_energy,
    compute_energy_calibration,
)
from .adjcluster import compute_adjcl_basics, compute_adjcl_advanced
from .adjopflash import compute_adjopflash_basics
from .ophit import compute_ophit_basic, compute_ophit_event
from .opflash import (
    compute_opflash_basic,
    compute_opflash_event,
    compute_opflash_main,
    compute_opflash_advanced,
    compute_opflash_matching,
)
from .drift import compute_true_drift
from .log import create_workflow_log


def compute_reco_workflow(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    workflow: Optional[str] = None,
    rm_branches: bool = False,
    debug: bool = False,
    verbose: Optional[Union[int, str]] = None,
    max_log_lines: Optional[int] = None,
) -> dict[str, dict]:
    """
    Compute the reco variables for the events in the run.
    All functions are called in the order they are defined in this file.
    All functions get the same arguments.

    Args:
        run: dictionary containing the run data.
        configs: dictionary containing the path to the configuration files for each geometry.
        params: dictionary containing the parameters for the reco functions.
        workflow: string containing the reco workflow to be used.
        rm_branches: boolean to remove the branches used in the reco workflow.
        debug: print debug information.
        verbose: global verbosity for workflow logs (0=warnings/errors, 1=log, 2=info).
        max_log_lines: maximum number of log lines to keep in terminal output.

    Returns:
        run (dict): dictionary containing the TTree with the new branches.
    """
    # Compute reco variables
    new_branches = []
    output = create_workflow_log(verbose=verbose, max_lines=max_log_lines)
    output += f"[green]Computing {workflow} workflow...[/green]"

    if workflow is None:
        rprint(f"No workflow selected. Returning same run.\n")
        return run

    is_truth_or_opflash = any(work in workflow for work in ["TRUTH", "OPFLASH"])
    is_edep = "EDEP" in workflow
    is_raw = "RAW" in workflow
    is_vertexing = "VERTEXING" in workflow
    is_marley = "MARLEY" in workflow
    is_track = "TRACK" in workflow
    is_adjcl = "ADJCL" in workflow
    is_ophit = "OPHIT" in workflow
    is_opflash = "OPFLASH" in workflow
    is_adjflash = "ADJFLASH" in workflow
    is_matchedflash = "MATCHEDFLASH" in workflow
    is_correction = "CORRECTION" in workflow
    is_calibration = "CALIBRATION" in workflow
    is_discrimination = "DISCRIMINATION" in workflow
    is_reconstruction = "RECONSTRUCTION" in workflow
    is_smearing = "SMEARING" in workflow
    is_analysis = "ANALYSIS" in workflow
    is_significance = "SIGNIFICANCE" in workflow

    is_precompute_workflow = any(
        [
            is_raw,
            is_vertexing,
            is_marley,
            is_adjcl,
            is_matchedflash,
            is_correction,
            is_calibration,
            is_discrimination,
            is_reconstruction,
            is_smearing,
            is_analysis,
            is_significance,
        ]
    )
    is_corr_or_calib_or_disc = any([is_correction, is_calibration, is_discrimination])
    is_corr_chain = any(
        [
            is_correction,
            is_calibration,
            is_discrimination,
            is_reconstruction,
            is_smearing,
            is_analysis,
            is_significance,
        ]
    )
    is_calib_chain = any(
        [
            is_calibration,
            is_discrimination,
            is_reconstruction,
            is_smearing,
            is_analysis,
            is_significance,
        ]
    )
    is_disc_chain = any(
        [
            is_discrimination,
            is_reconstruction,
            is_smearing,
            is_analysis,
            is_significance,
        ]
    )
    is_reco_chain = any(
        [is_reconstruction, is_smearing, is_analysis, is_significance]
    )
    is_smear_chain = any([is_smearing, is_analysis, is_significance])

    if is_truth_or_opflash:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches

    elif is_edep:
        run, output, this_new_branches = compute_signal_energies(
            run,
            configs,
            params,
            trees=["Truth"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )

    elif is_precompute_workflow:
        run, output, this_new_branches = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = split_vector_branches(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = update_default_values(
            run,
            configs,
            params=params,
            trees=(
                ["Truth", "Reco"] if workflow in ["ANALYSIS", "SIGNIFICANCE"] else None
            ),
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        run, output, this_new_branches = compute_true_drift(
            run,
            configs,
            params=params,
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches

    if workflow == "SIGNIFICANCE":
        run, output, new_branches_surface = compute_particle_surface(
            run,
            configs,
            params,
            ["Truth", "Reco"],
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += new_branches_surface
        run, output, this_new_branches = compute_particle_weights(
            run,
            configs,
            params,
            rm_branches=rm_branches,
            output=output,
            debug=debug,
        )
        new_branches += this_new_branches

    if is_marley:
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

    elif is_raw:
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

    elif is_vertexing:
        run, output, this_new_branches = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches
        # run, output, this_new_branches = compute_cluster_adjrecox(
        #     run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        # )
        # new_branches += this_new_branches

    elif is_track:
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

    elif is_adjcl:
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
    elif is_ophit:
        run, output, this_new_branches = compute_ophit_basic(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        run, output, this_new_branches = compute_ophit_event(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches

    elif is_opflash:
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

    elif is_adjflash:
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

    elif is_matchedflash:
        run, output, this_new_branches = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, output=output, debug=debug
        )
        new_branches += this_new_branches

    ### CALIBRATION AND RECONSTRUCTION WORKFLOWS
    elif is_corr_chain:
        trees = ["Reco"]
        if is_corr_or_calib_or_disc:
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
        if is_corr_chain:
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
        if is_calib_chain:
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
                clusters=(
                    ["", "Electron"]
                    if is_corr_or_calib_or_disc
                    else [""]
                ),
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches
        # if workflow in ["DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if is_disc_chain:
            run, output, this_new_branches = compute_cluster_calibration(
                run,
                configs,
                params,
                clusters=(
                    ["", "Electron"]
                    if is_corr_or_calib_or_disc
                    else [""]
                ),
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
        if is_reco_chain:
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
        if is_smear_chain:
            run, output, this_new_branches = compute_energy_calibration(
                run,
                configs,
                params,
                rm_branches=rm_branches,
                output=output,
                debug=debug,
            )
            new_branches += this_new_branches

    rendered_log = str(output)
    if rendered_log.strip():
        rprint(rendered_log)
    rprint(f"[green]{workflow} workflow completed!\n[/green]")
    if debug:
        rprint(f"New branches: {new_branches}")
    return run
