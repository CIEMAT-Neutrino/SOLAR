import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
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
        ROOT / 'data' / 'analysis' / 'daynight-json',
    ]


def analysis_json_globs(analysis, filename_pattern):
    pnfs_dir = Path('/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR') / str(analysis).upper()
    globs = [str(pnfs_dir / '*' / '*' / 'marley' / filename_pattern)]
    for base_dir in analysis_json_search_dirs(analysis):
        globs.append(str(base_dir / '*' / '*' / 'marley' / filename_pattern))
    return globs


def default_pdf_export_enabled():
    return True


def export_marp_pdf(markdown_path):
    md_path = Path(markdown_path)
    pdf_path = md_path.with_suffix('.pdf')
    errors = []

    def _short_reason(message):
        if message is None:
            return 'unknown error'
        text = str(message).replace('\r', '\n')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return 'unknown error'
        preferred_prefixes = [
            'Failed converting Markdown.',
            'No suitable browser found.',
            'Neither',
            'Permission denied',
            'File is not a zip file',
            'Command failed with exit code',
        ]
        for prefix in preferred_prefixes:
            for line in lines:
                if line.startswith(prefix):
                    return line
        for line in lines:
            if line.startswith('[ ERROR ]'):
                return line.replace('[ ERROR ]', '').strip()
        return lines[0]

    def _ensure_path_access(path, is_dir=False):
        p = Path(path)
        if not p.exists():
            return False
        try:
            p.chmod(0o755)
        except OSError:
            pass
        mode = os.R_OK | os.X_OK
        return os.access(str(p), mode)

    def _valid_executable(path):
        p = Path(path)
        if not p.exists() or not p.is_file():
            return False
        try:
            if p.stat().st_size <= 0:
                return False
        except OSError:
            return False
        _ensure_path_access(p)
        return os.access(str(p), os.X_OK)

    def _tools_dir():
        env_dir = os.environ.get('SOLAR_TOOLS_DIR', '').strip()
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.append(ROOT / '.tools')
        candidates.append(Path.home() / '.cache' / 'solar-tools')
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            _ensure_path_access(candidate, is_dir=True)
            if os.access(str(candidate), os.W_OK | os.X_OK):
                return candidate
        return ROOT / '.tools'

    def _run_export(command):
        try:
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        except OSError as exc:
            return None, _short_reason(str(exc))
        if result.returncode == 0:
            return pdf_path, None
        error = (result.stderr or result.stdout or '').strip()
        if not error:
            error = f'Command failed with exit code {result.returncode}'
        return None, _short_reason(error)

    def _discover_local_marp():
        tools_dir = _tools_dir()
        candidates = sorted(tools_dir.glob('marp-cli-v*-linux*/marp'))
        valid = [candidate for candidate in candidates if _valid_executable(candidate)]
        return str(valid[-1]) if valid else None

    def _discover_local_npx():
        tools_dir = _tools_dir()
        candidates = sorted(tools_dir.glob('node-v*-linux-x64/bin/npx'))
        valid = [candidate for candidate in candidates if _valid_executable(candidate)]
        return str(valid[-1]) if valid else None

    def _discover_local_browser():
        tools_dir = _tools_dir()
        patterns = [
            'chrome-for-testing-*/chrome-linux64/chrome',
            'chrome-linux64/chrome',
            'chromium*/chrome',
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(sorted(tools_dir.glob(pattern)))
        valid = [candidate for candidate in candidates if _valid_executable(candidate)]
        return str(valid[-1]) if valid else None

    def _bootstrap_local_marp():
        existing = _discover_local_marp()
        if existing is not None:
            return existing, None
        tools_dir = _tools_dir()
        tools_dir.mkdir(parents=True, exist_ok=True)
        _ensure_path_access(tools_dir, is_dir=True)
        release_url = 'https://api.github.com/repos/marp-team/marp-cli/releases/latest'
        try:
            with urllib.request.urlopen(release_url, timeout=20) as response:
                release = json.loads(response.read().decode('utf-8'))
        except Exception as exc:  # noqa: BLE001
            return None, f'Could not resolve marp-cli release: {exc}'
        linux_asset = None
        for asset in release.get('assets', []):
            if str(asset.get('name', '')).endswith('-linux.tar.gz'):
                linux_asset = asset
                break
        if linux_asset is None:
            return None, 'No Linux marp-cli asset found in latest release'
        archive = tools_dir / str(linux_asset.get('name'))
        extract_dir = tools_dir / str(linux_asset.get('name')).replace('.tar.gz', '')
        try:
            if not archive.exists() or archive.stat().st_size == 0:
                with urllib.request.urlopen(str(linux_asset.get('browser_download_url')), timeout=120) as response, open(archive, 'wb') as out:
                    out.write(response.read())
            extract_dir.mkdir(parents=True, exist_ok=True)
            _ensure_path_access(extract_dir, is_dir=True)
            with tarfile.open(archive, 'r:gz') as tar_in:
                tar_in.extractall(path=extract_dir)
            marp_path = extract_dir / 'marp'
            if not _valid_executable(marp_path):
                return None, f'Standalone marp binary is not executable at {marp_path}'
            return str(marp_path), None
        except Exception as exc:  # noqa: BLE001
            return None, f'Failed to bootstrap standalone marp-cli: {exc}'

    def _bootstrap_local_browser():
        existing = _discover_local_browser()
        if existing is not None:
            return existing, None
        tools_dir = _tools_dir()
        tools_dir.mkdir(parents=True, exist_ok=True)
        _ensure_path_access(tools_dir, is_dir=True)
        info_url = 'https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json'
        try:
            with urllib.request.urlopen(info_url, timeout=30) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except Exception as exc:  # noqa: BLE001
            return None, f'Could not resolve Chrome-for-Testing version: {exc}'
        stable = payload.get('channels', {}).get('Stable', {})
        version = str(stable.get('version', '')).strip()
        downloads = stable.get('downloads', {}).get('chrome', [])
        linux_download = None
        for item in downloads:
            if str(item.get('platform')) == 'linux64':
                linux_download = item
                break
        if linux_download is None:
            return None, 'No linux64 Chrome-for-Testing download found'
        archive_name = f'chrome-for-testing-{version}-linux64.zip' if version else 'chrome-for-testing-linux64.zip'
        archive_path = tools_dir / archive_name
        extract_dir = tools_dir / (f'chrome-for-testing-{version}' if version else 'chrome-for-testing')

        def _download_archive():
            curl_bin = shutil.which('curl')
            if curl_bin is not None:
                result = subprocess.run(
                    [curl_bin, '-fL', str(linux_download.get('url')), '-o', str(archive_path)],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    error = (result.stderr or result.stdout or '').strip()
                    raise RuntimeError(error or 'Failed to download Chrome-for-Testing archive with curl')
                return
            with urllib.request.urlopen(str(linux_download.get('url')), timeout=120) as response, open(archive_path, 'wb') as out:
                out.write(response.read())

        try:
            if not archive_path.exists() or archive_path.stat().st_size == 0:
                _download_archive()
            extract_dir.mkdir(parents=True, exist_ok=True)
            _ensure_path_access(extract_dir, is_dir=True)
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_in:
                    zip_in.extractall(path=extract_dir)
            except zipfile.BadZipFile:
                if archive_path.exists():
                    archive_path.unlink()
                _download_archive()
                with zipfile.ZipFile(archive_path, 'r') as zip_in:
                    zip_in.extractall(path=extract_dir)
            browser = extract_dir / 'chrome-linux64' / 'chrome'
            if not _valid_executable(browser):
                return None, f'Chrome binary is not executable at {browser}'
            return str(browser), None
        except Exception as exc:  # noqa: BLE001
            return None, f'Failed to bootstrap Chrome-for-Testing: {exc}'

    def _latest_lts_node_version():
        index_url = 'https://nodejs.org/dist/index.json'
        with urllib.request.urlopen(index_url, timeout=20) as response:
            payload = json.loads(response.read().decode('utf-8'))
        for entry in payload:
            if entry.get('lts') and 'linux-x64' in entry.get('files', []):
                return str(entry.get('version'))
        raise RuntimeError('No LTS linux-x64 Node.js release found')

    def _bootstrap_local_npx():
        existing = _discover_local_npx()
        if existing is not None:
            return existing, None
        curl_bin = shutil.which('curl')
        tar_bin = shutil.which('tar')
        if curl_bin is None or tar_bin is None:
            return None, "Missing required tools to bootstrap Node.js ('curl' and 'tar')"
        tools_dir = _tools_dir()
        tools_dir.mkdir(parents=True, exist_ok=True)
        _ensure_path_access(tools_dir, is_dir=True)
        try:
            node_version = _latest_lts_node_version()
        except Exception as exc:  # noqa: BLE001
            return None, f'Could not resolve latest LTS Node.js version: {exc}'
        node_folder = f'node-{node_version}-linux-x64'
        install_dir = tools_dir / node_folder
        if not install_dir.exists():
            archive = tools_dir / f'{node_folder}.tar.xz'
            if not archive.exists() or archive.stat().st_size == 0:
                url = f'https://nodejs.org/dist/{node_version}/{node_folder}.tar.xz'
                download = subprocess.run([curl_bin, '-fsSL', url, '-o', str(archive)], cwd=str(ROOT), capture_output=True, text=True)
                if download.returncode != 0:
                    error = (download.stderr or download.stdout or '').strip()
                    return None, error or 'Failed to download Node.js archive'
            extract = subprocess.run([tar_bin, '-xJf', str(archive), '-C', str(tools_dir)], cwd=str(ROOT), capture_output=True, text=True)
            if extract.returncode != 0:
                error = (extract.stderr or extract.stdout or '').strip()
                return None, error or 'Failed to extract Node.js archive'
        for bin_name in ['node', 'npx', 'npm', 'corepack']:
            bin_path = install_dir / 'bin' / bin_name
            if bin_path.exists():
                _ensure_path_access(bin_path)
        npx_path = install_dir / 'bin' / 'npx'
        if _valid_executable(npx_path):
            return str(npx_path), None
        return None, f'Bootstrapped Node.js but npx not found at {npx_path}'

    marp_script = ROOT / 'scripts' / 'marp-pdf.sh'
    if marp_script.exists():
        out_pdf, err = _run_export([str(marp_script), str(md_path), str(pdf_path)])
        if out_pdf is not None:
            return out_pdf, None
        if err:
            errors.append(f'scripts/marp-pdf.sh: {err}')

    marp_bin = shutil.which('marp')
    if marp_bin is not None:
        out_pdf, err = _run_export([marp_bin, str(md_path), '--pdf', '-o', str(pdf_path)])
        if out_pdf is not None:
            return out_pdf, None
        if err:
            errors.append(f'marp: {err}')

    local_marp = _discover_local_marp()
    if local_marp is None:
        local_marp, local_marp_error = _bootstrap_local_marp()
        if local_marp_error is not None:
            errors.append(f'bootstrap standalone marp: {_short_reason(local_marp_error)}')
    if local_marp is not None:
        out_pdf, err = _run_export([local_marp, str(md_path), '--pdf', '-o', str(pdf_path)])
        if out_pdf is None and err and 'permission denied' in err.lower():
            try:
                retry_marp = Path(tempfile.gettempdir()) / 'solar-marp-cli'
                shutil.copy2(local_marp, retry_marp)
                _ensure_path_access(retry_marp)
                out_pdf, err = _run_export([str(retry_marp), str(md_path), '--pdf', '-o', str(pdf_path)])
            except OSError as exc:
                err = _short_reason(exc)
        if out_pdf is not None:
            return out_pdf, None
        if err and 'no suitable browser found' in err.lower():
            browser_path, browser_error = _bootstrap_local_browser()
            if browser_path is not None:
                out_pdf, err = _run_export([local_marp, str(md_path), '--pdf', '--browser-path', browser_path, '-o', str(pdf_path)])
                if out_pdf is not None:
                    return out_pdf, None
            elif browser_error is not None:
                errors.append(f'bootstrap browser: {_short_reason(browser_error)}')
        if err:
            errors.append(f'standalone marp: {err}')

    npx_bin = shutil.which('npx')
    if npx_bin is None:
        npx_bin, bootstrap_error = _bootstrap_local_npx()
        if bootstrap_error is not None:
            errors.append(f'bootstrap npx: {_short_reason(bootstrap_error)}')
    if npx_bin is not None:
        out_pdf, err = _run_export([npx_bin, '-y', '@marp-team/marp-cli', str(md_path), '--pdf', '-o', str(pdf_path)])
        if out_pdf is not None:
            return out_pdf, None
        if err and 'no suitable browser found' in err.lower():
            browser_path, browser_error = _bootstrap_local_browser()
            if browser_path is not None:
                out_pdf, err = _run_export([
                    npx_bin,
                    '-y',
                    '@marp-team/marp-cli',
                    str(md_path),
                    '--pdf',
                    '--browser-path',
                    browser_path,
                    '-o',
                    str(pdf_path),
                ])
                if out_pdf is not None:
                    return out_pdf, None
            elif browser_error is not None:
                errors.append(f'bootstrap browser: {_short_reason(browser_error)}')
        if err:
            errors.append(f'npx @marp-team/marp-cli: {err}')

    if errors:
        return None, ' | '.join(errors)
    return None, "Neither 'scripts/marp-pdf.sh', 'marp', nor 'npx' is available"
