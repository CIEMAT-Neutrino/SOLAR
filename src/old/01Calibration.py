# ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/data/ .

# Load function libraries and set up the environment
from lib.__init__ import *
import sys

sys.path.insert(0, "../")

np.seterr(divide="ignore", invalid="ignore")
plt.rcParams.update({"font.size": 15})

# Load macro configuration
default_dict = {}
user_input = initialize_macro(
    "00Process",
    ["config_file", "root_file", "rewrite", "debug"],
    default_dict=default_dict,
    debug=True,
)
user_input = check_macro_config(user_input, debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]
configs = {config: config + "_config"}
names = {config: user_input["root_file"]}

run = load_multi(
    names,
    configs,
    load_all=False,
    preset="CALIBRATION",
    debug=False,
)

# Load analysis configuration
analysis_info = read_input_file(
    "analysis",
    INTEGERS=["RECO_ENERGY_RANGE", "RECO_ENERGY_BINS",
              "NADIR_RANGE", "NADIR_BINS"],
    debug=False,
)
energy_edges = np.linspace(
    analysis_info["RECO_ENERGY_RANGE"][0],
    analysis_info["RECO_ENERGY_RANGE"][1],
    analysis_info["RECO_ENERGY_BINS"] + 1,
)
energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2
bin_width = energy_edges[1] - energy_edges[0]

# Compute the calibration workflow
run = compute_reco_workflow(run, configs, workflow="CALIBRATION", debug=True)

# Filter the data for calibration
max_energy = 20
acc = 50
info = json.load(
    open("../config/" + f"{config}/{config}_config" + ".json", "r"))
total_energy_filter = run["Reco"]["SignalParticleE"] < max_energy * 1e-3
# electron_filter     = run["Reco"]["MarleyFrac"][:,0] > 0.9
geo_filter = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0]
version_filter = np.asarray(run["Reco"]["Version"]) == info["VERSION"][0]
time_filter = abs(run["Reco"]["Time"]) < info["EVENT_TICKS"][0]
neutron_filter = (run["Reco"]["TMarleyPDG"][:, :] != 2112).all(axis=1)

filter1 = (
    (total_energy_filter)
    * (geo_filter)
    * (version_filter)
    * (neutron_filter)
    * (time_filter)
    * (run["Reco"]["Primary"])
)

# Plot the calibration workflow
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Time Profile", "Correction", "Charge Calibration")
)
if user_input["debug"]:
    print_colored("Filtering %d events" % np.sum(filter1), "SUCCESS")
fig, corr_popt, perr = get_hist2d_fit(
    np.abs(run["Reco"]["Time"][filter1]),
    (run["Reco"]["Charge"] / (1e3 * run["Reco"]["TMarleyE"][:, 2]))[filter1],
    acc,
    fig,
    1,
    1,
    func_type="exponential",
    debug=user_input["debug"],
)

# Save the true energy fit parameters to a txt file
if not os.path.exists("../config/" + config + "/" + config + "_calib/"):
    os.makedirs("../config/" + config + "/" + config + "_calib/")
with open(
    "../config/"
    + config
    + "/"
    + config
    + "_calib/"
    + config
    + "_charge_correction.txt",
    "w",
) as f:
    f.write("CHARGE_AMP: %f\n" % corr_popt[0])
    f.write("ELECTRON_TAU: %f\n" % corr_popt[1])
plt.close()

run = compute_cluster_energy(
    run,
    configs,
    params={"DEFAULT_ENERGY_TIME": "Time",
            "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime"},
    debug=user_input["debug"],
)

fig.add_trace(
    go.Histogram2d(
        x=run["Reco"]["Time"][filter1],
        y=(
            run["Reco"]["Charge"]
            * run["Reco"]["Correction"]
            / (corr_popt[0] * 1e3 * run["Reco"]["TMarleyE"][:, 2])
        )[filter1],
        coloraxis="coloraxis",
        nbinsx=acc,
        nbinsy=acc,
    ),
    row=1,
    col=2,
)

fig, reco_popt, perr = get_hist2d_fit(
    1e3 * run["Reco"]["TMarleyE"][:, 2][filter1],
    run["Reco"]["Energy"][filter1],
    acc,
    fig,
    2,
    1,
    func_type="linear",
    debug=user_input["debug"],
)
fig, res_popt, perr = get_hist1d_fit(
    (
        run["Reco"]["Charge"]
        * run["Reco"]["Correction"]
        / (corr_popt[0] * 1e3 * run["Reco"]["TMarleyE"][:, 2])
    )[filter1],
    2 * acc,
    fig,
    2,
    2,
    func_type="gauss",
    debug=user_input["debug"],
)

fig.update_layout(
    coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Counts")),
    showlegend=False,
    title="Calibration",
    xaxis1_title="Time [ticks]",
    xaxis2_title="Time [ticks]",
    xaxis3_title="True Electron Energy [MeV]",
    xaxis4_title="Corr. Charge/Energy [Norm]",
    yaxis1_title="Charge/Energy [ADC x ticks/MeV]",
    yaxis2_title="Corr. Charge/Energy [Norm]",
    yaxis3_title="Reco Electron Energy [MeV]",
    yaxis4_title="Norm.",
)

fig = format_coustom_plotly(fig, fontsize=18)
if not os.path.exists("../images/calibration/%s_calibration/" % config):
    os.makedirs("../images/calibration/%s_calibration/" % config)
fig.write_image(
    "../images/calibration/%s_calibration/%s_calibration.png" % (
        config, config),
    width=2400,
    height=1080,
)
print_colored(
    "-> Saved images to ../images/calibration/%s_calibration/%s_calibration.png"
    % (config, config),
    "SUCCESS",
)
fig.show()
