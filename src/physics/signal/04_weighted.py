import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/solar/nhits"
data_path = f"{root}/output/data/solar/nhits"

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--signal", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the background folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--analysis",
    type=str,
    default="Sensitivity",
    help="Analysis type for PNFS best-cuts resolution (Weighted_Distributions_Fiducial only).",
)
parser.add_argument("--nhits",  type=int, default=None, help="NHits cut override (else from PNFS)")
parser.add_argument("--ophits", type=int, default=None, help="OpHits cut override (else from PNFS)")
parser.add_argument("--adjcls", type=int, default=None, help="AdjCl cut override (else from PNFS)")
parser.add_argument("--mc_filter_threshold", type=int, default=2, help="Min unweighted MC events per energy bin; bins below are zeroed (matches 03_analysis.py default).")

parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help="Oscillation weighting backend. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)

args = parser.parse_args()
config = args.config
name = args.signal
configs = {config: [name]}

if not os.path.exists(f"{save_path}/{args.folder.lower()}"):
    os.makedirs(f"{save_path}/{args.folder.lower()}")

user_input = {
    "workflow": "SIGNIFICANCE",
    "weights": {
        "marley": [
            "SignalParticleWeight",
            "SignalParticleWeightb8",
            "SignalParticleWeighthep",
        ],
        "neutron": ["SignalParticleWeight"],
        "gamma": ["SignalParticleWeight"],
        "alpha": ["SignalParticleWeight"],
        "radiological": ["SignalParticleWeight"],
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
        "radiological": ["radiological"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
        "radiological": ["rgb(120, 94, 240)"],
    },
    "yzoom": {"marley": [0, 6], "neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8], "radiological": [0, 6]},
    "rewrite": True,
    "debug": True,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
)

run = compute_reco_workflow(
    run,
    configs,
    params=(
        {
            "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
            "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
            "PARTICLE_TYPE": "signal",
            "PARTICLE_WEIGHTING": "volume",
            "OSCILLATION_BACKEND": args.oscillation_backend,
        }
        if "marley" in args.signal
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    rm_branches=False,
    workflow=user_input["workflow"],
    debug=args.debug,
)


# ── Best-cuts helpers (for Weighted_Distributions_Fiducial only) ───────────────

def _extract_best_cuts(obj, cfg, sig):
    key = (cfg, sig, "SolarEnergy")
    if isinstance(obj, dict):
        row = obj.get(key)
        if row is None:
            return None
        return int(row["NHits"]), int(row["OpHits"]), int(row["AdjCl"])
    if key not in obj.columns:
        return None
    s = obj[key]
    return int(s["NHits"]), int(s["OpHits"]), int(s["AdjCl"])


def _load_best_cuts(cfg, sig, analysis, folder, info):
    _base = f"{info['PATH']}/{analysis.upper()}/{folder.lower()}/{cfg}/{sig}"
    _pkl  = f"{_base}/{cfg}_{sig}_highest_{analysis}.pkl"
    if not os.path.exists(_pkl):
        rprint(f"[yellow][WARNING][/yellow] Best-cuts pkl not found: {_pkl}. Cut stage will be skipped.")
        return None
    return _extract_best_cuts(pickle.load(open(_pkl, "rb")), cfg, sig)


# ── Threshold-scan helpers ─────────────────────────────────────────────────────

def _scan_ge(vals, wts, thresholds):
    """sum(wts[vals >= t]) and count(vals >= t) for each t in thresholds."""
    if len(vals) == 0:
        z = np.zeros(len(thresholds))
        return z, z.astype(int)
    order = np.argsort(vals)
    sv, sw = vals[order], wts[order]
    cs = np.concatenate([[0.0], np.cumsum(sw)])
    idx = np.searchsorted(sv, thresholds, side="left")
    return cs[-1] - cs[idx], (len(vals) - idx).astype(int)


def _scan_lt(vals, wts, thresholds):
    """sum(wts[vals < t]) and count(vals < t) for each t in thresholds."""
    if len(vals) == 0:
        z = np.zeros(len(thresholds))
        return z, z.astype(int)
    order = np.argsort(vals)
    sv, sw = vals[order], wts[order]
    cs = np.concatenate([[0.0], np.cumsum(sw)])
    idx = np.searchsorted(sv, thresholds, side="left")
    return cs[idx], idx.astype(int)


def _mc_event_filter(energy_all, mask, rb_edges, threshold):
    """Per-event bool (length = sum(mask)): True if energy bin has >= threshold MC events.

    Mirrors 03_analysis.py: mc_counts computed at rebinned (1 MeV) level; bins below
    threshold are zeroed. Applied only to weighted Counts, not NEvents.
    """
    _rb_bins = len(rb_edges) - 1
    e = energy_all[mask]
    idx = np.digitize(e, rb_edges) - 1
    valid = (idx >= 0) & (idx < _rb_bins)
    mc_counts = np.bincount(idx[valid], minlength=_rb_bins)
    mc_filter_rb = mc_counts >= threshold
    event_ok = np.zeros(len(e), dtype=bool)
    event_ok[valid] = mc_filter_rb[idx[valid]]
    return event_ok


# ── Main loop ──────────────────────────────────────────────────────────────────

nhits_list          = []   # full per-event distributions (original schema)
nhits_fiducial_list = []   # threshold-scan summary per (stage, hit variable)

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    _params_path = f"{root}/config/{config}/{config}_params.json"
    _exposure    = float(json.loads(open(_params_path).read()).get("EVALUATION_EXPOSURE_YEARS", 20.0)) if os.path.exists(_params_path) else 20.0
    _scale       = get_full_detector_mass(config, info) * _exposure   # kT·yr → events/raw_weight

    for name in configs[config]:

        # ── Resolve best cuts (fiducial summary only) ──────────────────────────
        _nhits, _ophits, _adjcls = args.nhits, args.ophits, args.adjcls
        if None in (_nhits, _ophits, _adjcls):
            _cuts = _load_best_cuts(config, name, args.analysis, args.folder, info)
            if _cuts is None and "marley" not in name:
                _cuts = _load_best_cuts(config, "marley", args.analysis, args.folder, info)
            if _cuts is not None:
                _nhits, _ophits, _adjcls = _cuts
                rprint(f"[cyan][INFO][/cyan] Best cuts for {config}/{name}: NHits={_nhits} OpHits={_ophits} AdjCl={_adjcls}")

        _has_cuts = None not in (_nhits, _ophits, _adjcls)
        _reco     = run["Reco"]
        _fid_mask = _reco["SignalParticleSurface"] < 3
        _all_mask = np.ones(len(_reco["Event"]), dtype=bool)

        # Global threshold ranges (full dataset so all stages share the same x-axis)
        _t_nhits  = np.arange(0, int(np.max(_reco["NHits"]))               + 2)
        _t_ophits = np.arange(0, int(np.max(_reco["MatchedOpFlashNHits"])) + 2)
        _t_adjcls = np.arange(0, int(np.max(_reco["AdjClNum"]))            + 2)

        _weights_list = list(zip(
            user_input["weights"][name.split("_")[0]],
            user_input["weight_labels"][name.split("_")[0]],
            user_input["colors"][name.split("_")[0]],
        ))
        _surfaces = [None, -1, 0, 1, 2, 3, 4] if name in ["marley", "neutron"] else [None]

        # ── Per-surface entries — original schema (unchanged) ──────────────────
        for (weight, weight_labels, color), surface in track(
            product(
                _weights_list,
                _surfaces,
            ),
            total=(3 if "marley" in args.signal else 1),
            description=f"Processing {name} - {config}",
        ):
            if surface is None:
                if args.folder.lower() in ["reduced", "truncated"]:
                    mask = _reco["SignalParticleSurface"] < 3
                else:
                    mask = _all_mask
            else:
                mask = _reco["SignalParticleSurface"] == surface

            if np.sum(mask) > 0:
                nhits_list.append(
                    {
                        "Config": config,
                        "Name": name,
                        "Folder": args.folder,
                        "Component": weight_labels,
                        "Weight": weight,
                        "Type": "signal" if "marley" in name else "background",
                        "Surface": surface,
                        "#Hits": _reco["NHits"][mask],
                        "#OpHits": _reco["MatchedOpFlashNHits"][mask],
                        "#AdjCls": _reco["AdjClNum"][mask],
                        "Counts": _reco[weight][mask],
                        "TrueEnergy": _reco["SignalParticleK"][mask],
                        "RecoEnergy": _reco["SolarEnergy"][mask],
                    }
                )
            else:
                if args.debug:
                    print(f"No events found for {name} - {config} - {weight} - Surface {surface}")

        # ── Fiducial threshold-scan summary ────────────────────────────────────
        # Stages: (name, nhits_base, ophits_base, adjcls_base, nc, oc, ac)
        # nhits_base / ophits_base / adjcls_base = base mask for each hit-variable scan.
        # For Raw and Fiducial all three bases are the same.
        # For NHits+OpHits+AdjCl each base fixes the other two cuts at best values.
        _scan_stages: list[tuple] = [
            ("Raw",      _all_mask, _all_mask, _all_mask, None, None, None),
            ("Fiducial", _fid_mask, _fid_mask, _fid_mask, None, None, None),
        ]
        if _has_cuts:
            _nhv = _reco["NHits"]
            _opv = _reco["MatchedOpFlashNHits"]
            _acv = _reco["AdjClNum"]
            _scan_stages.append((
                "NHits+OpHits+AdjCl",
                _fid_mask & (_opv >= _ophits) & (_acv < _adjcls),   # scan NHits
                _fid_mask & (_nhv >= _nhits)  & (_acv < _adjcls),   # scan OpHits
                _fid_mask & (_nhv >= _nhits)  & (_opv >= _ophits),  # scan AdjCls
                _nhits, _ophits, _adjcls,
            ))
        else:
            rprint(f"[yellow][WARNING][/yellow] No cut values for {config}/{name}. NHits+OpHits+AdjCl stage skipped.")

        # Precompute per-event mc_filter for each (stage, scan_var) — weight-independent.
        # Uses SolarEnergy + hep_rebin (1 MeV bins, same as 03_analysis.py).
        _energy_for_filter = _reco["SolarEnergy"]
        _mc_filters = [
            (
                _mc_event_filter(_energy_for_filter, m_nh, hep_rebin, args.mc_filter_threshold),
                _mc_event_filter(_energy_for_filter, m_op, hep_rebin, args.mc_filter_threshold),
                _mc_event_filter(_energy_for_filter, m_ac, hep_rebin, args.mc_filter_threshold),
            )
            for _, m_nh, m_op, m_ac, *_ in _scan_stages
        ]

        for weight, weight_labels, color in _weights_list:
            wts = _reco[weight]
            for (stage_name, m_nhits, m_ophits, m_adjcls, nc, oc, ac), (mcf_nh, mcf_op, mcf_ac) in zip(_scan_stages, _mc_filters):
                _meta = {
                    "Config":      config,
                    "Name":        name,
                    "Folder":      args.folder,
                    "Component":   weight_labels,
                    "Weight":      weight,
                    "Type":        "signal" if "marley" in name else "background",
                    "Stage":       stage_name,
                    "NHits":       nc,
                    "OpHits":      oc,
                    "AdjCl":       ac,
                    "Exposure":    _exposure,
                    "CountsUnit":  f"events / {_exposure:.0f} yr",
                }

                # NEvents uses raw unweighted counts (no mc_filter); Counts zeroes low-stat bins.
                _nh_vals = _reco["NHits"][m_nhits]
                c, n = _scan_ge(_nh_vals[mcf_nh], wts[m_nhits][mcf_nh], _t_nhits)
                _, n_raw = _scan_ge(_nh_vals, wts[m_nhits], _t_nhits)
                nhits_fiducial_list.append({**_meta, "#Hits": _t_nhits,  "#OpHits": np.nan, "#AdjCls": np.nan, "Counts": c * _scale, "NEvents": n_raw})

                _op_vals = _reco["MatchedOpFlashNHits"][m_ophits]
                c, n = _scan_ge(_op_vals[mcf_op], wts[m_ophits][mcf_op], _t_ophits)
                _, n_raw = _scan_ge(_op_vals, wts[m_ophits], _t_ophits)
                nhits_fiducial_list.append({**_meta, "#Hits": np.nan, "#OpHits": _t_ophits, "#AdjCls": np.nan, "Counts": c * _scale, "NEvents": n_raw})

                _ac_vals = _reco["AdjClNum"][m_adjcls]
                c, n = _scan_lt(_ac_vals[mcf_ac], wts[m_adjcls][mcf_ac], _t_adjcls)
                _, n_raw = _scan_lt(_ac_vals, wts[m_adjcls], _t_adjcls)
                nhits_fiducial_list.append({**_meta, "#Hits": np.nan, "#OpHits": np.nan, "#AdjCls": _t_adjcls, "Counts": c * _scale, "NEvents": n_raw})

# ── Save ───────────────────────────────────────────────────────────────────────

save_df(
    pd.DataFrame(nhits_list),
    f"{data_path}",
    config=config,
    name=name,
    filename=f"Weighted_Distributions",
    subfolder=f"{args.folder.lower()}",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

if nhits_fiducial_list:
    save_df(
        pd.DataFrame(nhits_fiducial_list),
        f"{data_path}",
        config=config,
        name=name,
        filename=f"Weighted_Distributions_Fiducial_{args.analysis}",
        subfolder=f"{args.folder.lower()}/{args.analysis.lower()}",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
