"""
cleanup_oscillation_templates.py — Remove stale oscillation template files
===================================================================

Scans PNFS directories for sensitivity signal/background templates and removes
files that do NOT correspond to the current OSCILLATION_GRID configuration.

Additionally, cleans up old best-cut files that were saved at non-standard paths
(e.g., {PATH}/SENSITIVITY/{folder}/{config}/{name}/...) and moves them to the
standard path {PATH}/SENSITIVITY/{config}/{name}/{folder}/...).

This ensures consistency between:
- The configured oscillation parameter grid
- The actual template files on disk
- The path structure (config/signal/folder/...)

Usage:
------
# Dry run (default - only reports what would be removed)
python3 src/tools/cleanup_oscillation_templates.py --config hd_1x2x6_centralAPA --signal marley

# Actually remove files
python3 src/tools/cleanup_oscillation_templates.py --config hd_1x2x6_centralAPA --signal marley --folder Truncated --force

# Clean specific energy
python3 src/tools/cleanup_oscillation_templates.py --config hd_1x2x6_centralAPA --signal marley --energy SolarEnergy --force

# Verbose output
python3 src/tools/cleanup_oscillation_templates.py --config hd_1x2x6_centralAPA --signal marley --verbose --dry-run

Options:
--------
  --config CONFIG      Detector configuration (default: hd_1x2x6_centralAPA)
  --signal SIGNAL      Signal name (default: marley)
  --folder FOLDER      Results folder(s): Reduced, Truncated, Nominal (default: all)
  --energy ENERGY      Energy variable(s) (default: SolarEnergy)
  --dry-run            Only report what would be removed, don't delete (default: True)
  --force             Actually remove files (use with caution!)
  --verbose           Show detailed output

What gets cleaned:
-----------------
1. Signal template files with oscillation parameters not in OSCILLATION_GRID
2. Old best-cut files at non-standard paths (folder/config/name/ -> config/name/folder/)
"""

import os
import sys
import re
import json
import argparse
from glob import glob

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from lib import *
    from lib.oscillation import make_oscillation_grid, load_analysis_info
except Exception as e:
    print(f"[WARNING] Could not import lib modules: {e}")
    print("This script requires the SOLAR library to be available.")
    sys.exit(1)

# Ensure rprint is available
try:
    from rich import print as rprint
except ImportError:
    # Fallback to regular print if rich is not available
    rprint = print


def parse_template_filename(filename: str) -> dict:
    """
    Extract oscillation parameters from a template filename.
    
    Expected format:
    {config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl
    
    Returns dict with keys: config, name, nhits, adjcl, ophits, dm2, sin13, sin12
    Returns None if filename doesn't match expected pattern.
    """
    # Remove .pkl extension
    basename = filename.replace('.pkl', '')
    
    # Pattern to match the full template filename
    # Example: hd_1x2x6_centralAPA_marley_NHits4_AdjCl10_OpHits4_dm2_6.000e-05_sin13_2.200e-02_sin12_3.040e-01
    pattern = r'^(.+)_NHits(\d+)_AdjCl(\d+)_OpHits(\d+)_dm2_([\d.e+-]+)_sin13_([\d.e+-]+)_sin12_([\d.e+-]+)$'
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    try:
        return {
            'config': match.group(1).rsplit('_', 1)[0],  # Handle config_name
            'name': match.group(1).rsplit('_', 1)[1] if '_' in match.group(1) else None,
            'nhits': int(match.group(2)),
            'adjcl': int(match.group(3)),
            'ophits': int(match.group(4)),
            'dm2': float(match.group(5)),
            'sin13': float(match.group(6)),
            'sin12': float(match.group(7)),
        }
    except (ValueError, IndexError):
        return None


def parse_background_filename(filename: str) -> dict:
    """
    Extract cut parameters from a background template filename.
    
    Expected format:
    {config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl
    
    Returns dict with keys: config, nhits, adjcl, ophits
    Returns None if filename doesn't match expected pattern.
    """
    basename = filename.replace('.pkl', '')
    
    # Pattern: hd_1x2x6_centralAPA_background_NHits4_AdjCl10_OpHits4
    pattern = r'^(.+)_background_NHits(\d+)_AdjCl(\d+)_OpHits(\d+)$'
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    try:
        return {
            'config': match.group(1),
            'nhits': int(match.group(2)),
            'adjcl': int(match.group(3)),
            'ophits': int(match.group(4)),
        }
    except (ValueError, IndexError):
        return None


def get_current_grid_points(analysis_info: dict) -> tuple[set, set, set]:
    """
    Get the current configured oscillation grid points from OSCILLATION_GRID.
    
    Returns three sets: (dm2_set, sin13_set, sin12_set)
    """
    dm2_list, sin13_list, sin12_list = make_oscillation_grid(analysis_info)
    return set(dm2_list), set(sin13_list), set(sin12_list)


def find_stale_signal_templates(path: str, current_dm2: set, current_sin13: set, current_sin12: set) -> list[dict]:
    """
    Find signal template files that don't match the current grid configuration.
    
    Returns list of dicts with file info and reason for being stale.
    """
    stale_files = []
    pattern = f"{path}/*.pkl"
    
    for filepath in glob(pattern):
        filename = os.path.basename(filepath)
        parsed = parse_template_filename(filename)
        
        if parsed is None:
            # Can't parse filename - might be a different type of file
            continue
        
        # Check if the oscillation parameters match the current grid
        dm2_ok = parsed['dm2'] in current_dm2
        sin13_ok = parsed['sin13'] in current_sin13
        sin12_ok = parsed['sin12'] in current_sin12
        
        if not (dm2_ok and sin13_ok and sin12_ok):
            reason = []
            if not dm2_ok:
                reason.append(f"dm2={parsed['dm2']:.3e} not in grid")
            if not sin13_ok:
                reason.append(f"sin13={parsed['sin13']:.3e} not in grid")
            if not sin12_ok:
                reason.append(f"sin12={parsed['sin12']:.3e} not in grid")
            
            stale_files.append({
                'filepath': filepath,
                'filename': filename,
                'parsed': parsed,
                'reason': ", ".join(reason)
            })
    
    return stale_files


def find_stale_results(path: str, current_dm2: set, current_sin13: set, current_sin12: set) -> list[dict]:
    """
    Find result files (DataFrames) that don't match the current grid configuration.
    
    Result files include: solar_sin12_df, solar_sin13_df, react_sin12_df, react_sin13_df, etc.
    These are saved in the results/ subdirectory.
    
    Returns list of dicts with file info and reason for being stale.
    """
    stale_files = []
    
    # Look for result directories
    result_dirs = glob(f"{path}/results/*/")
    
    for result_dir in result_dirs:
        # Look for DataFrame pkl files in the result directory
        df_pattern = f"{result_dir}/*_df*.pkl"
        
        for filepath in glob(df_pattern):
            filename = os.path.basename(filepath)
            # Extract parameters from filename
            # Format: {config}_{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_{df_type}{tag}.pkl
            # But we need the dm2, sin13, sin12 from the directory structure
            
            # The result files are in: .../results/{profile_name}/
            # They don't encode oscillation parameters in the filename
            # Instead, they should be in a directory structure that matches the grid
            
            # For result files, we check if they're in a directory that corresponds to
            # a valid profile. Result files themselves don't have oscillation parameters
            # in their names, so we can't easily determine if they're stale.
            # 
            # Actually, result files are per-cut-combination, not per-oscillation-point.
            # They contain the full chi2 grid. So they're not directly tied to specific
            # oscillation points. The stale check doesn't apply here the same way.
            
            # For now, skip result file cleanup as it's more complex
            pass
    
    return stale_files


def main():
    parser = argparse.ArgumentParser(
        description="Remove stale oscillation template files that don't match current OSCILLATION_GRID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hd_1x2x6_centralAPA",
        help="Detector configuration"
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="marley",
        help="Signal name"
    )
    parser.add_argument(
        "--folder",
        nargs="*",
        type=str,
        choices=["Reduced", "Truncated", "Nominal"],
        default=["Reduced", "Truncated", "Nominal"],
        help="Result folders to clean (default: all)"
    )
    parser.add_argument(
        "--energy",
        nargs="*",
        type=str,
        choices=["SignalParticleK", "MainK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
        default=None,
        help="Energy variables to clean (default: all configured in physics.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only report what would be removed, don't delete (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Actually remove files (use with caution!)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Override dry-run if force is specified
    if args.force:
        args.dry_run = False
    
    # Load analysis info
    analysis_info = load_analysis_info(str(root))
    
    # Get current grid points
    current_dm2, current_sin13, current_sin12 = get_current_grid_points(analysis_info)
    
    if args.verbose:
        rprint(f"[cyan][INFO][/cyan] Current OSCILLATION_GRID:")
        rprint(f"  dm2 points: {len(current_dm2)} (range: {min(current_dm2):.3e} to {max(current_dm2):.3e})")
        rprint(f"  sin13 points: {len(current_sin13)} (range: {min(current_sin13):.3f} to {max(current_sin13):.3f})")
        rprint(f"  sin12 points: {len(current_sin12)} (range: {min(current_sin12):.3f} to {max(current_sin12):.3f})")
    
    # Load config info
    config_path = f"{root}/config/{args.config}/{args.config}_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    info = json.loads(open(config_path).read())
    base_path = info['PATH']
    
    # Determine energy variables to process
    if args.energy is None or len(args.energy) == 0:
        # Default to SolarEnergy which is the standard for Sensitivity analysis
        energies = ["SolarEnergy"]
    else:
        energies = args.energy
    
    # Track statistics
    total_stale = 0
    total_size = 0
    
    # Process each folder
    for folder in args.folder:
        folder_lower = folder.lower()
        
        # Process signal templates
        signal_path = f"{base_path}/SENSITIVITY/{args.config}/{args.signal}/{folder_lower}"
        
        if args.verbose:
            rprint(f"\n[cyan][INFO][/cyan] Scanning signal templates in: {signal_path}")
        
        if os.path.exists(signal_path):
            for energy in energies:
                energy_path = f"{signal_path}/{energy}"
                if not os.path.exists(energy_path):
                    if args.verbose:
                        rprint(f"  Skipping non-existent: {energy_path}")
                    continue
                
                # Check for template_suffix subdirectories (e.g., for charge threshold variants)
                for item in sorted(os.listdir(energy_path)):
                    item_path = f"{energy_path}/{item}"
                    if os.path.isdir(item_path):
                        # This is a template_suffix directory (e.g., "_10yr", "_charge100")
                        if args.verbose:
                            rprint(f"  Scanning directory: {item}")
                        stale = find_stale_signal_templates(item_path, current_dm2, current_sin13, current_sin12)
                        target_dir = item_path
                    elif item.endswith('.pkl'):
                        # Direct pkl file in energy directory
                        if args.verbose:
                            rprint(f"  Scanning energy directory: {energy_path}")
                        stale = find_stale_signal_templates(energy_path, current_dm2, current_sin13, current_sin12)
                        # Filter to only files directly in this energy directory
                        stale = [s for s in stale if os.path.dirname(s['filepath']) == energy_path]
                        target_dir = energy_path
                    else:
                        continue
                    
                    if stale:
                        total_stale += len(stale)
                        for s in stale:
                            try:
                                total_size += os.path.getsize(s['filepath'])
                            except OSError:
                                # File might have been deleted or inaccessible
                                pass
                        
                        if args.dry_run:
                            rprint(f"\n[yellow][DRY-RUN][/yellow] Would remove {len(stale)} stale signal template(s) from: {target_dir}")
                            for s in stale[:5]:  # Show first 5 to avoid flooding output
                                rprint(f"  - {s['filename']}: {s['reason']}")
                            if len(stale) > 5:
                                rprint(f"  ... and {len(stale) - 5} more")
                        else:
                            rprint(f"\n[green][REMOVING][/green] Removing {len(stale)} stale signal template(s) from: {target_dir}")
                            for s in stale:
                                try:
                                    os.remove(s['filepath'])
                                    if args.verbose:
                                        rprint(f"  ✓ Removed: {s['filename']}")
                                except Exception as e:
                                    rprint(f"  ✗ Failed to remove {s['filename']}: {e}")
        
        # Process background templates
        background_path = f"{base_path}/SENSITIVITY/{args.config}/background/{folder_lower}"
        
        if args.verbose:
            rprint(f"\n[cyan][INFO][/cyan] Scanning background templates in: {background_path}")
        
        if os.path.exists(background_path):
            for energy in energies:
                energy_path = f"{background_path}/{energy}"
                if not os.path.exists(energy_path):
                    continue
                
                # Background templates don't have oscillation parameters in filename
                # They are: {config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl
                # So they're not directly tied to specific oscillation points
                # We can't determine if they're stale based on the grid
                
                if args.verbose:
                    rprint(f"  Background templates: skipping (not directly tied to oscillation grid)")
        
        # Clean up old best-cut files at non-standard paths
        old_best_cut_pattern = f"{base_path}/SENSITIVITY/{folder_lower}/{args.config}/{args.signal}/"
        if args.verbose:
            rprint(f"\n[cyan][INFO][/cyan] Checking for old best-cut files at: {old_best_cut_pattern}")
        
        old_best_cut_dir = old_best_cut_pattern
        if os.path.exists(old_best_cut_dir):
            # Look for highest_SENSITIVITY*.pkl files
            best_cut_files = glob(f"{old_best_cut_dir}/{args.config}_{args.signal}_highest_SENSITIVITY*.pkl")
            
            if best_cut_files:
                total_stale += len(best_cut_files)
                for s in best_cut_files:
                    try:
                        total_size += os.path.getsize(s)
                    except OSError:
                        pass
                
                if args.dry_run:
                    rprint(f"\n[yellow][DRY-RUN][/yellow] Would remove {len(best_cut_files)} old best-cut file(s) from non-standard path:")
                    for s in best_cut_files[:5]:
                        rprint(f"  - {os.path.basename(s)}")
                    if len(best_cut_files) > 5:
                        rprint(f"  ... and {len(best_cut_files) - 5} more")
                else:
                    rprint(f"\n[green][REMOVING][/green] Removing {len(best_cut_files)} old best-cut file(s) from non-standard path:")
                    for s in best_cut_files:
                        try:
                            os.remove(s)
                            if args.verbose:
                                rprint(f"  ✓ Removed: {os.path.basename(s)}")
                        except Exception as e:
                            rprint(f"  ✗ Failed to remove {os.path.basename(s)}: {e}")
    
    # Summary
    rprint(f"\n{'='*70}")
    rprint(f"[cyan][SUMMARY][/cyan]")
    rprint(f"  Total stale signal templates found: {total_stale}")
    rprint(f"  Total size: {total_size / (1024*1024):.2f} MB" if total_size > 0 else "  Total size: 0 B")
    rprint(f"  Mode: {'DRY-RUN (no files removed)' if args.dry_run else 'FORCE (files removed)'}")
    
    if args.dry_run and total_stale > 0:
        rprint(f"\n  To actually remove these files, run with --force")


if __name__ == "__main__":
    main()
