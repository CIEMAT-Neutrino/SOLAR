import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal"

# KDE files are folder-independent: the oscillation-weighted energy PDF is a
# theoretical quantity (solar flux × P_osc) unaffected by fiducialization.
# Folder-specific effects (surface cuts, reduction factors) apply to backgrounds
# at analysis time — the same KDE pkl is reused across all folder variants.
parser = argparse.ArgumentParser(
    description="Pre-generate per-nadir_slice oscillation-weighted signal energy KDEs"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()

os.makedirs(data_path, exist_ok=True)

configs = {args.config: [args.name]}
user_input = {
    "rewrite": args.rewrite,
    "debug": args.debug,
}

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = 1e9  # kT
    eff_flux_b8 = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["b8"]
    )
    eff_flux_hep = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["hep"]
    )
    eff_flux_comb = get_detected_solar_spectrum(
        bins=energy_centers, mass=detector_mass, components=["b8", "hep"]
    )
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)

    (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
        dm2=None,
        sin13=None,
        sin12=None,
        path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
        ext="pkl",
        auto=True,
        debug=user_input["debug"],
    )

    for dm2, sin13, sin12 in track(zip(dm2_list, sin13_list, sin12_list), description="Processing oscillation parameters", total=len(dm2_list)):
        truth_oscillation_df = pd.read_pickle(
            f"{info['PATH']}/data/OSCILLATION/pkl/raw/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
        )
        for name, (eff_flux, label) in product(
            configs[config],
            zip((eff_flux_b8, eff_flux_hep, eff_flux_comb), ["b8", "hep", "comb"]),
        ):

            for nadir_slice in truth_oscillation_df.index[: len(truth_oscillation_df) // 2]:
                this_nadir_flux = (
                    eff_flux
                    * 60
                    * 60
                    * 24
                    * 365
                    * np.array(list(truth_oscillation_df.loc[nadir_slice].values))
                )
                kde_nadir = KernelDensity(
                    bandwidth=ebin, kernel="gaussian", algorithm="kd_tree"
                ).fit(energy_centers[:, None], sample_weight=this_nadir_flux)
                save_pkl(
                    kde_nadir,
                    f"{data_path}/osc/nadir_{-nadir_slice:.3f}",
                    config,
                    None,
                    filename=f"{name}_{label}_kde_p_nadir_{-nadir_slice:.3f}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )
                if user_input["debug"]:
                    print(
                        f"KDE model saved for nadir: {nadir_slice:.3f}, dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}"
                    )