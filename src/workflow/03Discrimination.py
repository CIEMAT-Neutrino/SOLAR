import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/workflow/discrimination"
data_path = f"{root}/data/workflow/discrimination"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_signal"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "DISCRIMINATION",
    "rewrite": True,
    "debug": True,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "Time",
        "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime",
    },
    workflow=user_input["workflow"],
    debug=user_input["debug"],
)

filtered_run, mask, output = compute_filtered_run(
    run, configs, presets=[user_input["workflow"]], debug=user_input["debug"]
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
acc = get_default_acc(len(data["Generator"]))
per = (3, 97)
fit = {
    "color": "grey",
    "func": "linear",
    "opacity": 0,
    "print": False,
    "show": True,
}
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        selected_list = []
        true_list = []

        for particle in ["Electron", "Gamma", "Neutron", "Alpha", "Proton", "Neutrino"]:
            true_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "SignalParticleK": data["SignalParticleK"],
                    "Particle": particle,
                    "Energy": data[f"{particle}K"],
                }
            )

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Electron", "Electron Energy Offset")
        )

        fit["spec_type"] = "intercept"
        fit["threshold"] = 0.4
        fig, popt_int, perr_int = get_hist2d_fit(
            data["SignalParticleK"],
            data["ElectronK"],
            fig=fig,
            idx=(1, 1),
            per=None,
            acc=acc,
            fit=fit,
            nanz=True,
            zoom=True,
        )

        intercept_list, count_list = get_hist1d_diff(
            data["SignalParticleK"], data["ElectronK"], -Particle.from_pdgid(11).mass
        )

        peaks, values = find_peaks(count_list, height=0.02)
        values["peak_heights"] = np.array(values["peak_heights"])

        fig.add_trace(
            go.Scatter(
                x=intercept_list,
                y=count_list,
                mode="lines",
                line=dict(color="grey"),
                line_shape="hvh",
            ),
            row=1,
            col=2,
        )
        # Add peaks to the plot
        fig.add_trace(
            go.Scatter(
                x=np.array(intercept_list)[peaks],
                y=np.array(values["peak_heights"]),
                mode="markers",
                marker=dict(color="red", size=10),
            ),
            row=1,
            col=2,
        )

        mid_popt = [1, -5]
        upper_filter = intercept_list[peaks] < -mid_popt[1]
        lower_filter = intercept_list[peaks] > -mid_popt[1]

        popt_int[0] = -np.sum(
            np.multiply(intercept_list[peaks], values["peak_heights"]),
            where=upper_filter,
        ) / np.sum(values["peak_heights"], where=upper_filter)

        popt_int[1] = -np.sum(
            np.multiply(intercept_list[peaks], values["peak_heights"]),
            where=lower_filter,
        ) / np.sum(values["peak_heights"], where=lower_filter)
        popt_int[1] = -intercept_list[peaks][
            np.argmax(values["peak_heights"] * lower_filter)
        ]

        def upper_func(x):
            return x - popt_int[0]

        def lower_func(x):
            return x - popt_int[1]

        upper_idx = np.where(
            data["ElectronK"] > data["SignalParticleK"] * mid_popt[0] + mid_popt[1]
        )
        lower_idx = np.where(
            data["ElectronK"] < data["SignalParticleK"] * mid_popt[0] + mid_popt[1]
        )
        data["Upper"] = np.asarray(
            data["ElectronK"] > data["SignalParticleK"] * mid_popt[0] + mid_popt[1],
            dtype=bool,
        )
        data["Lower"] = np.asarray(
            data["ElectronK"] < data["SignalParticleK"] * mid_popt[0] + mid_popt[1],
            dtype=bool,
        )

        for intercept, label, pos in zip(
            popt_int, ["Upper", "Lower"], ["bottom left", "bottom right"]
        ):
            fig.add_vline(
                x=-intercept,
                row=1,
                col=2,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"{label}:\n {-intercept:.2f}",
                annotation_position=pos,
            )

        fig.add_vline(x=-mid_popt[1], row=1, col=2, line_color="black")

        fig = format_coustom_plotly(
            fig, matches=(None, None), title="Neutrino Energy Reconstruction"
        )
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm")),
            showlegend=False,
            xaxis1_title="True Neutrino Energy (MeV)",
            yaxis1_title="True Electron Energy (MeV)",
            xaxis2_title="Energy Offset (MeV)",
            yaxis2_title="Norm",
        )

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"True_Electron_Energy",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        new_params = params.copy()
        if info["BACKGROUND"]:
            # Create a plot for the adjcl radius and energy distribution to evaluate the best separation between signal and background
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "AdjCl Distance Distribution",
                    "AdjCl Energy Distribution",
                ),
            )
            # Create index for entries != 0 and purity > 0
            signal_idx = np.where((data["AdjClCharge"] != 0) & (data["AdjClPur"] > 0))
            background_idx = np.where(
                (data["AdjClCharge"] != 0) & (data["AdjClPur"] == 0)
            )
            # Export the limits to cut on the adjcl radius and energy maximizing the purity
            signal_over_background_ratio = 0
            for radius_limit, energy_limit in product(
                np.arange(1, 100, 2), np.arange(1, 100, 2)
            ):
                adjcl_radius_limit = np.percentile(
                    data["AdjClR"][background_idx], radius_limit
                )
                adjcl_energy_limit = np.percentile(
                    data["AdjClCharge"][background_idx], energy_limit
                )
                # Print the number of entries in the signal and background samples that pass the limits
                selected_signal_idx = np.where(
                    (data["AdjClR"][signal_idx] < adjcl_radius_limit)
                    + (data["AdjClCharge"][signal_idx] > adjcl_energy_limit)
                )
                selected_background_idx = np.where(
                    (data["AdjClR"][background_idx] < adjcl_radius_limit)
                    + (data["AdjClCharge"][background_idx] > adjcl_energy_limit)
                )
                if (
                    len(data["AdjClR"][signal_idx][selected_signal_idx])
                    / np.sqrt(
                        len(data["AdjClR"][background_idx][selected_background_idx])
                    )
                    > signal_over_background_ratio
                ):
                    signal_over_background_ratio = len(
                        data["AdjClR"][signal_idx][selected_signal_idx]
                    ) / np.sqrt(
                        len(data["AdjClR"][background_idx][selected_background_idx])
                    )
                    adjcl_radius_limit_best = adjcl_radius_limit
                    adjcl_energy_limit_best = adjcl_energy_limit
                    selected_signal_idx_best = selected_signal_idx
                    selected_background_idx_best = selected_background_idx

            rprint(
                f"AdjCl Distance Limit: {adjcl_radius_limit_best:.2f} cm, AdjCl Charge Limit: {adjcl_energy_limit_best:.2f} ADC x ticks"
            )
            rprint(
                f"Selected Signal Entries: {100*len(data['AdjClR'][signal_idx][selected_signal_idx_best]) / len(data['AdjClR'][signal_idx]):.2f}% \nSelected Background Entries: {100*len(data['AdjClR'][background_idx][selected_background_idx_best]) / len(data['AdjClR'][background_idx]):.2f}%"
            )

            for i, idx in enumerate([signal_idx, background_idx]):
                fig.add_trace(
                    go.Histogram2dContour(
                        x=data["AdjClR"][idx].flatten(),
                        y=data["AdjClCharge"][idx].flatten(),
                        coloraxis="coloraxis",
                        colorbar=dict(title="Norm"),
                        contours=dict(
                            coloring="heatmap", showlabels=True, labelfont_size=12
                        ),
                        zmin=1,
                        zmax=None,  # Set zmax to None to allow for white regions
                        colorscale=[
                            [0, "white"],
                            [1, "blue"],
                        ],  # Change 'blue' to your desired color
                    ),
                    row=1,
                    col=1 if idx is signal_idx else 2,
                )

                selected_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "AdjClR": data["AdjClR"][idx],
                        "AdjClCharge": data["AdjClCharge"][idx],
                        "LimitR": adjcl_radius_limit_best,
                        "LimitCharge": adjcl_energy_limit_best,
                        "Signal": i == 0,
                    }
                )

            # Draw the limits on the plots
            fig.add_vline(
                x=adjcl_radius_limit_best,
                line_dash="dash",
                line_color="red",
                # annotation_text=f"Radius Limit: {adjcl_radius_limit:.2f} cm",
                # annotation_position="top left",
            )
            fig.add_hline(
                y=adjcl_energy_limit_best,
                line_dash="dash",
                line_color="red",
                # annotation_text=f"Energy Limit: {adjcl_energy_limit:.2f} MeV",
                # annotation_position="top right",
            )
            fig.update_xaxes(title_text="Adj. Cluster Distance (cm)", row=1, col=1)
            fig.update_yaxes(title_text="Adj. Cluster Charge", row=1, col=1)
            fig.update_yaxes(title_text="Adj. Cluster Distance (cm)", row=1, col=2)
            fig = format_coustom_plotly(
                fig,
                matches=(None, None),
                title="Adj. Cluster Charge vs Distance",
            )
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"AdjCl_Radius_Charge_Distribution",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            new_params["MIN_BKG_R"] = adjcl_radius_limit_best
            new_params["MAX_BKG_CHARGE"] = adjcl_energy_limit_best

            update_json_file(f"{root}/config/{config}/{config}_params.json", new_params)

        else:
            rprint(
                f"Production does not contain background. Selecting all adjacent clusters as true signal."
            )

        run, output, branches = compute_total_energy(
            run,
            configs,
            params=new_params,
            rm_branches=False,
            output=output,
            debug=user_input["debug"],
        )

        features = get_default_info(root, "ML_FEATURES")
        # Create the dataframe and load the feature branches
        df = npy2df(
            {"Data": data},
            "Data",
            branches=features
            + ["Primary", "Generator", "SignalParticleK", "NHits", "Upper", "Lower"],
            debug=user_input["debug"],
        )

        # Display the dataframe
        A = df[(df["Upper"] == True) + (df["Generator"] != 1)]
        B = df[(df["Lower"] == True) & (df["Generator"] == 1)]

        A_train = A.sample(frac=0.5)
        A_test = A.drop(A_train.index)
        B_train = B.sample(frac=0.5)
        B_test = B.drop(B_train.index)

        train_df = pd.concat([A_train, B_train])
        train_df["Label"] = np.where(train_df["Upper"] != True, 0, 1)

        test_df = pd.concat([A_test, B_test])
        test_df["Label"] = np.where(test_df["Upper"] != True, 0, 1)

        # Create a random forest classifier object
        rf_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_split=2, min_samples_leaf=1
        )

        # Train the classifier on the training data
        rf_classifier.fit(train_df[features], train_df["Label"])
        # Use the trained classifier to predict the labels for the test data
        test_df["ML"] = rf_classifier.predict(test_df[features])

        # Print the accuracy of the classifier
        # print(
        #     f"Random Forest Accuracy: {accuracy_score(test_df['Label'], test_df['ML']):.2f}"
        # )

        # Check if the path to the model file exists
        if not os.path.exists(f"{root}/config/{config}/{name}/models/"):
            # If the path does not exist, create it
            os.makedirs(f"{root}/config/{config}/{name}/models/")
        # Save the trained classifier to a file
        with open(
            f"{root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl",
            "wb",
        ) as model_file:
            pickle.dump(rf_classifier, model_file)

        # Check if the model file already exists
        if os.path.exists(
            f"{root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl"
        ):
            # If the model file exists, load the model from the file
            with open(
                f"{root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl",
                "rb",
            ) as model_file:
                rf_classifier = pickle.load(model_file)
        else:
            # If the model file does not exist, print an error message
            print("Model file not found")

        # Genarate canvas with 2 subplots
        fig, axs = plt.subplots(1, 3, figsize=(26, 6))
        for df in [train_df, test_df]:
            rf_probabilities = rf_classifier.predict_proba(df[features])[:, 1]
            # Compute the false positive rate, true positive rate, and thresholds
            fpr, tpr, thresholds = roc_curve(df["Label"], rf_probabilities)
            # Compute the purity and efficiency
            purity = tpr
            efficiency = tpr / (tpr + fpr)
            axs[0].plot(
                efficiency,
                purity,
                color="mediumblue" if df is train_df else "firebrick",
                ls="-",
                lw=2,
            )

        # Add the remaining code for the first plot
        axs[0].set_title("Purity vs Efficiency")
        axs[0].set_xlabel("Efficiency")
        axs[0].set_ylabel("Purity")
        axs[0].legend(["Train", "Test"])

        # Plot histogram of random forest scores for signal and background
        axs[1].hist(
            rf_classifier.predict_proba(test_df[features])[:, 1][test_df["Label"] == 1],
            density=True,
            bins=50,
            color="mediumblue",
            alpha=0.8,
            label="Upper",
        )
        axs[1].hist(
            rf_classifier.predict_proba(test_df[features])[:, 1][test_df["Label"] == 0],
            density=True,
            bins=50,
            color="firebrick",
            alpha=0.8,
            label="Lower",
        )
        axs[1].set_xlabel("Random Forest Score")
        axs[1].set_ylabel("Density")
        axs[1].set_title("Random Forest")
        axs[1].legend()

        # Find the ideal threshold according to the best compromise between true positive rate and false positive rate
        maximal_difference = np.argmax(tpr - fpr)
        threshold = thresholds[maximal_difference]

        print(f"Best threshold: {threshold:.2f}")
        colored_line(fpr, tpr, thresholds, axs[2], lw=4)
        # Plot diagonal line for reference
        axs[2].axvline(x=fpr[maximal_difference], color="grey", linestyle="--")
        axs[2].axhline(y=tpr[maximal_difference], color="grey", linestyle="--")
        axs[2].set_xlabel("False Positive Rate")
        axs[2].set_ylabel("True Positive Rate")
        axs[2].set_title("ROC Curve")
        axs[2].plot([0, 1], [0, 1], color="black", linestyle=":")
        # Add a legend to the plot including the ideal threshold and the AUC value
        axs[2].legend(
            ["ROC Curve", f"Threshold {threshold:.2f}", f"AUC {auc(fpr, tpr):.2f}"]
        )

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Random_Forest_Score",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        importance = rf_classifier.feature_importances_
        fig = make_subplots(cols=1, rows=1)
        fig.add_trace(
            go.Bar(x=features, y=importance, marker_color="blue", opacity=0.75)
        )
        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            margin={"auto": False, "margin": (100, 100, 100, 150)},
        )
        fig.update_layout(
            title=f"Random Forest Feature Importance - {config} {name}",
            xaxis_title="Features",
            yaxis_title="Importance",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Random_Forest_Feature_Importance",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Make a cross-correlation matrix
        corr = train_df[features].corr()

        # Create a heatmap figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis"
            )
        )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title=f"Feature Correlation Matrix - {config} {name}",
            figsize=(1200, 1000),
            margin={"auto": False, "margin": (150, 100, 100, 150)},
        )

        # Set the title and axis labels
        fig.update_layout(
            xaxis=dict(title="", tickangle=45),
            yaxis=dict(title="", tickangle=315),
        )

        # Show the figure
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Correlation_Matrix",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        df["ML"] = rf_classifier.predict(df[features])
        df["Discriminant"] = rf_classifier.predict_proba(df[features])[:, 1]
        # Compute reco energy
        upper_idx = df["Discriminant"] >= threshold
        lower_idx = df["Discriminant"] < threshold
        df.loc[upper_idx, "SolarEnergy"] = upper_func(df.loc[upper_idx, "Energy"])
        df.loc[lower_idx, "SolarEnergy"] = lower_func(df.loc[lower_idx, "Energy"])

        fig = make_subplots(rows=1, cols=1)
        for discriminant_idx, discriminant in enumerate(["Lower", "Upper"]):
            h, edges = np.histogram(
                df[df["Label"] == discriminant_idx]["Discriminant"],
                bins=50,
                range=(0, 1),
            )
            fig.add_trace(
                go.Scatter(
                    x=(edges[1:] + edges[:-1]) / 2,
                    y=h / np.sum(h),
                    mode="lines",
                    line_shape="hvh",
                    line=dict(
                        color=compare[discriminant_idx],
                        width=2,
                    ),
                )
            )

        fig.add_vline(
            threshold,
            line_dash="dot",
            line_color="grey",
            annotation_text=f"Threshold {threshold:.1f}",
        )
        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title="Cluster Discriminant",
            tickformat=(".2f", ".2f"),
        )

        fig.update_layout(
            bargap=0,
            barmode="overlay",
            legend_title="Discriminant Label",
            xaxis_title="Discriminant",
            yaxis_title="Probability",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Discriminant_Calibration_1D",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        popt_corr = {}
        perr_corr = {}
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Upper", "Lower"))

        fit["spec_type"] = "max"
        fit["bounds"] = ([0.1, -1], [1.1, 9])
        for idx, col, label in zip([upper_idx, lower_idx], [1, 2], ["Upper", "Lower"]):
            fit["threshold"] = (
                len(df["SignalParticleK"][idx]) * params["DISCRIMINATION_THRESHOLD"]
            )
            acc = get_default_acc(len(df["SignalParticleK"][idx]))
            fig, popt, perr = get_hist2d_fit(
                df["SignalParticleK"][idx],
                df["SolarEnergy"][idx],
                fig,
                idx=(1, col),
                per=per,
                acc=acc,
                fit=fit,
                density=False,
                zoom=True,
                debug=user_input["debug"],
            )
            if popt is not None:
                popt_corr[label] = popt
                perr_corr[label] = perr
            else:
                rprint(
                    f"Could not fit {label} energy calibration. Please check the data."
                )
                popt_corr[label] = [0, 0]
                perr_corr[label] = [0, 0]

        fig = format_coustom_plotly(fig, title="ML-Based Neutrino Energy Smearing")
        fig.update_yaxes(title="Reco Energy (MeV)", row=1, col=1)
        fig.update_xaxes(title="True Neutrino Energy (MeV)")
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Energy_Calibration_2D",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Save as json file
        if not os.path.exists(f"{root}/config/{config}/{name}/{config}_calib/"):
            os.makedirs(f"{root}/config/{config}/{name}/{config}_calib/")
        with open(
            f"{root}/config/{config}/{name}/{config}_calib/{config}_discriminant_calibration.json",
            "w",
        ) as f:
            json.dump(
                {
                    "DISCRIMINANT_THRESHOLD": mid_popt[1],
                    "ML_THRESHOLD": threshold,
                    "UPPER": {
                        "OFFSET": popt_int[0],
                        "ENERGY_AMP": popt_corr["Upper"][0],
                        "INTERSECTION": popt_corr["Upper"][1],
                    },
                    "LOWER": {
                        "OFFSET": popt_int[1],
                        "ENERGY_AMP": popt_corr["Lower"][0],
                        "INTERSECTION": popt_corr["Lower"][1],
                    },
                },
                f,
            )

        rprint(
            f"Saved reco energy fit parameters to: {root}/config/{config}/{name}/{config}_calib/{config}_{name}_discriminant_calibration.json"
        )
        for this_list, df_filename in zip(
            [true_list, selected_list], ["Neutrino_CC_Production", "AdjCl_Selection"]
        ):
            save_df(
                pd.DataFrame(this_list),
                data_path,
                config,
                name,
                None,
                filename=df_filename,
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
