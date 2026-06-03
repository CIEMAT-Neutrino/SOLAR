import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_ENERGY = "SolarEnergy"

STANDARD_CONFIGS = [
    ("hd_1x2x6_centralAPA", "HD Central"),
    ("hd_1x2x6_lateralAPA", "HD Lateral"),
    ("vd_1x8x14_3view_30deg_nominal", "VD Top"),
    ("vd_1x8x14_3view_30deg_shielded", "VD Bottom Shielded"),
]
CONFIG_ALIAS_MAP = {config: alias for config, alias in STANDARD_CONFIGS}


def config_alias(config_name):
    return CONFIG_ALIAS_MAP.get(config_name, config_name)


def output_energy_label(energy):
    return DEFAULT_ENERGY if energy == DEFAULT_ENERGY else energy


def energy_candidates(energy):
    if energy == DEFAULT_ENERGY:
        return ["SolarEnergy", "Solar"]
    return [energy]


def pick_most_recent(paths):
    existing = [path for path in paths if path is not None and path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_latest(base_dir, patterns):
    if not base_dir.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(base_dir.glob(pattern))
    return pick_most_recent(candidates)


def analysis_json_search_dirs(analysis):
    analysis_key = str(analysis).strip().upper()
    analysis_dir = analysis_key.lower()
    return [
        ROOT / 'data' / 'analysis' / 'best-sigma-json' / analysis_dir,
        ROOT / 'data' / 'analysis' / f'{analysis_dir}-json',
    ]


def analysis_json_globs(analysis, filename_pattern):
    pnfs_dir = Path('/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR') / str(analysis).upper()
    globs = [str(pnfs_dir / '*' / '*' / 'marley' / filename_pattern)]
    for base_dir in analysis_json_search_dirs(analysis):
        globs.append(str(base_dir / '*' / '*' / 'marley' / filename_pattern))
    return globs


_LAR_DENSITY_G_PER_CM3 = 1.396


def compute_fiducial_mass_kt(config, fid_x, fid_y, fid_z, lar_density=_LAR_DENSITY_G_PER_CM3):
    """Return fiducial LAr mass in kt for the full detector after applying fid cuts.

    Each production simulates ONE HALF of the physical module (one cathode/drift side).
    workspace × FULL_DETECTOR_FACTOR = full module, so a factor 0.5 is applied universally.

    X geometry mirrors build_fiducial_spatial_mask in lib_fiducial.py:
      - lateralAPA: half-drift sim → drift_factor=2 (both sides); net with 0.5 = ×1
      - centralAPA: full two-drift workspace → drift_factor=1; net with 0.5 = ×0.5
      - VD: one-drift workspace (top or bottom) → drift_factor=1; net with 0.5 = ×0.5
    """
    config_path = ROOT / "config" / config / f"{config}_config.json"
    if not config_path.exists():
        return None
    with open(config_path) as fh:
        info = json.load(fh)
    size_x = info["DETECTOR_SIZE_X"] + 2 * info.get("DETECTOR_GAP_X", 0)
    size_y = info["DETECTOR_SIZE_Y"] + 2 * info.get("DETECTOR_GAP_Y", 0)
    size_z = info["DETECTOR_SIZE_Z"] + 2 * info.get("DETECTOR_GAP_Z", 0)
    full_factor = info.get("FULL_DETECTOR_FACTOR", 1)
    fx = int(fid_x) if fid_x is not None else 0
    fy = int(fid_y) if fid_y is not None else 0
    fz = int(fid_z) if fid_z is not None else 0
    config_lower = str(config).lower()
    if config_lower == "hd_1x2x6_lateralapa":
        fid_x_size = size_x - fx
        drift_factor = 2  # workspace = one drift; real APA has two
    elif config_lower == "hd_1x2x6_centralapa":
        fid_x_size = size_x - 2 * fx
        drift_factor = 1
    else:  # VD: one-sided top-boundary fiducial (one drift only)
        fid_x_size = size_x - fx
        drift_factor = 1
    fid_y_size = size_y - 2 * fy
    fid_z_size = size_z - 2 * fz
    if fid_x_size <= 0 or fid_y_size <= 0 or fid_z_size <= 0:
        return 0.0
    # 0.5: each production covers half the physical module (one cathode side);
    # workspace × FULL_DETECTOR_FACTOR spans the full module.
    return fid_x_size * fid_y_size * fid_z_size * lar_density * drift_factor * full_factor * 0.5 / 1e9


def default_pdf_export_enabled():
    return True


def _find_local_chrome(tools_dir):
    if not tools_dir.exists():
        return None
    for pattern in [
        'chrome-linux64/chrome',
        'chrome-for-testing-*/chrome-linux64/chrome',
        'chromium*/chrome',
    ]:
        candidates = [p for p in sorted(tools_dir.glob(pattern)) if os.access(str(p), os.X_OK)]
        if candidates:
            return str(candidates[-1])
    return None


def export_marp_pdf(markdown_path):
    """Export a Marp markdown file to PDF via marp-pdf.sh.

    Returns (pdf_path, None) on success, (None, error_string) on failure.
    """
    md_path = Path(markdown_path)
    pdf_path = md_path.with_suffix('.pdf')

    print(f"Exporting PDF {md_path.name}...", flush=True)
    marp_script = ROOT / 'src' / 'tools' / 'marp_pdf.sh'
    if not marp_script.exists():
        return None, f'src/tools/marp_pdf.sh not found at {marp_script}'

    result = subprocess.run(
        [str(marp_script), str(md_path), str(pdf_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path, None

    # Extract error lines; prefer [ERROR]/fail lines, exclude pure [INFO]/[WARN] noise
    combined = ((result.stderr or '') + '\n' + (result.stdout or '')).strip()
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    error_lines = [ln for ln in lines if re.search(r'\[.*ERROR.*\]|fail', ln, re.IGNORECASE)]
    if not error_lines:
        error_lines = [ln for ln in lines if not re.search(r'\[.*INFO.*\]|\[.*WARN.*\]', ln)]
    if not error_lines:
        error_lines = lines[-3:] if lines else []
    return None, ' | '.join(error_lines) if error_lines else f'exit {result.returncode}'
