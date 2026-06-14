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


def analysis_json_globs(analysis, filename_pattern):
    analysis_key = str(analysis).strip().upper()
    analysis_dir = analysis_key.lower()
    pnfs_dir = Path('/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR') / analysis_key
    return [
        str(pnfs_dir / '*' / '*' / 'marley' / filename_pattern),
        str(ROOT / 'config' / '*' / f'best-sigma-json' / analysis_dir / '*' / '*' / filename_pattern),
        str(ROOT / 'config' / '*' / f'{analysis_dir}-json' / '*' / '*' / filename_pattern),
    ]


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


_ANALYSIS_DIR = {
    "daynight": "day-night",
    "hep": "hep",
    "sensitivity": "sensitivity",
}

# When a config has no oscillogram in the requested analysis, try these fallbacks in order.
_OSCILLOGRAM_FALLBACK_ORDER = ["sensitivity", "day-night", "hep"]


def gather_oscillogram_specs(folder, energy, analysis_name):
    analysis_lower = str(analysis_name).lower()
    analysis_dir = _ANALYSIS_DIR.get(analysis_lower, analysis_lower)
    energy_label = output_energy_label(energy)

    primary_root = ROOT / "output" / "images" / "analysis" / analysis_dir / "oscillogram"
    fallback_roots = [
        ROOT / "output" / "images" / "analysis" / _ANALYSIS_DIR.get(a, a) / "oscillogram"
        for a in _OSCILLOGRAM_FALLBACK_ORDER
        if _ANALYSIS_DIR.get(a, a) != analysis_dir
    ]

    specs = []
    for config_key, display_name in STANDARD_CONFIGS:
        osc = nadir_proj = nadir_weighted = signal_1d = None
        for osc_root in [primary_root] + fallback_roots:
            config_dir = osc_root / config_key / "marley" / folder
            if not config_dir.exists():
                continue
            if osc is None:
                osc = find_latest(config_dir, [f"{config_key}_marley_Oscillogram_{energy_label}.png"])
            if nadir_proj is None:
                nadir_proj = find_latest(config_dir, [f"{config_key}_marley_Oscillogram_NadirProjection_{energy_label}.png"])
            if nadir_weighted is None:
                nadir_weighted = find_latest(config_dir, [f"{config_key}_marley_Oscillogram_NadirWeighted_{energy_label}.png"])
            if signal_1d is None:
                signal_1d = find_latest(config_dir, [f"{config_key}_marley_Signal1D_{energy_label}_FidOnly.png"])
            if osc and nadir_proj:
                break
        specs.append({
            "name": display_name,
            "config": config_key,
            "folder": folder,
            "oscillogram": osc.relative_to(ROOT).as_posix() if osc else None,
            "nadir_projection": nadir_proj.relative_to(ROOT).as_posix() if nadir_proj else None,
            "nadir_weighted": nadir_weighted.relative_to(ROOT).as_posix() if nadir_weighted else None,
            "signal_1d": signal_1d.relative_to(ROOT).as_posix() if signal_1d else None,
        })
    return specs


def render_oscillogram_slides(specs, show_signal_1d=True):
    slides = []
    for spec in specs:
        osc = spec.get("oscillogram")
        nadir = spec.get("nadir_projection")
        nadir_w = spec.get("nadir_weighted")
        sig1d = spec.get("signal_1d") if show_signal_1d else None
        if osc or nadir:
            osc_block = f'    <img src="../../{osc}">' if osc else "    <p>Oscillogram not available.</p>"
            nadir_block = f'    <img src="../../{nadir}">' if nadir else "    <p>Nadir projection not available.</p>"
            parts = [
                f"### {spec['name']}",
                "",
                '<div class="two-col">',
                "  <div>",
                '    <p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>',
                osc_block,
                "  </div>",
                "  <div>",
                '    <p><strong>Nadir projection</strong></p>',
                nadir_block,
                "  </div>",
                "</div>",
            ]
            slides.append("\n".join(parts))
            if nadir_w or sig1d:
                extra = [f"### {spec['name']} (cont.)"]
                if nadir_w and sig1d:
                    extra += [
                        "",
                        '<div class="two-col">',
                        "  <div>",
                        '    <p><strong>Nadir-weighted oscillogram</strong></p>',
                        f'    <img src="../../{nadir_w}">',
                        "  </div>",
                        "  <div>",
                        '    <p><strong>1D fiducial signal spectrum</strong></p>',
                        f'    <img src="../../{sig1d}">',
                        "  </div>",
                        "</div>",
                    ]
                elif nadir_w:
                    extra += [
                        "",
                        '<div class="center">',
                        '  <p><strong>Nadir-weighted oscillogram</strong></p>',
                        f'  <img src="../../{nadir_w}">',
                        "</div>",
                    ]
                else:
                    extra += [
                        "",
                        '<div class="center">',
                        '  <p><strong>1D fiducial signal spectrum</strong></p>',
                        f'  <img src="../../{sig1d}">',
                        "</div>",
                    ]
                slides.append("\n".join(extra))
        else:
            slides.append(f"### {spec['name']}\n\nNo oscillogram found.")
    if not slides:
        return "### Oscillograms\n\nNo oscillogram PNGs found."
    return "\n\n---\n\n".join(slides)


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
