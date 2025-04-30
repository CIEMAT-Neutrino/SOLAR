import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/discrimination/"

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
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_signal"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

config = parser.parse_args().config
name = parser.parse_args().name

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
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

filtered_run, mask, output = compute_filtered_run(
    run, configs, presets=[user_input["workflow"]], debug=user_input["debug"]
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
acc = 100
fit = {
    "color": "grey",
    "func": "linear",
    "opacity": 0,
    "print": True,
    "range": (0, 10),
    "show": False,
    "spec_type": "max",
    "threshold": 0.7,
    "trimm": (10, 10),
}
for config in configs:
    for name in configs[config]:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Electron", "Electron Energy Offset")
        )

        fit["spec_type"] = "intercept"
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

        features = get_default_info(root, "ML_FEATURES")
        # Create the dataframe and load the feature branches
        df = npy2df(
            {"Data": data},
            "Data",
            branches=features
            + ["Primary", "Generator", "SignalParticleK", "NHits", "Upper", "Lower"],
            debug=user_input["debug"],
        )
        # List all columns in the dataframe
        print(df.columns)
        print(df["TotalAdjClEnergy"])

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
        print(
            f"Random Forest Accuracy: {accuracy_score(test_df['Label'], test_df['ML']):.2f}"
        )

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
        print(importance)
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
            title="Random Forest: Feature Importance",
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
            title="Correlation Matrix",
            figsize=(1200, 1000),
            margin={"auto": False, "margin": (150, 100, 100, 150)},
        )

        # Set the title and axis labels
        fig.update_layout(
            xaxis=dict(title="Features", tickangle=45),
            yaxis=dict(title="Features", tickangle=315),
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

        # this_df = df[
        #     (df["Primary"] == True)
        #     & (df["Generator"] == 1)
        #     & (df["NHits"] > 2)
        #     & (df["SignalParticleK"] < 20)
        # ]

        fig = px.histogram(
            df, x="Discriminant", histnorm="probability", opacity=0.75, color="Label"
        )
        fig.add_vline(
            threshold,
            line_dash="dot",
            line_color="grey",
            annotation_text=f" Discriminat Threshold {threshold:.2f}",
        )
        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title="Cluster Discriminant",
            tickformat=(".2f", ".2f"),
        )

        fig.update_layout(
            barmode="overlay",
            bargap=0,
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

        # this_df = df[
        #     (df["Primary"] == True) & (df["Generator"] == 1) & (df["NHits"] > 1)
        # ]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Upper", "Lower"))
        popt_corr = []
        perr_corr = []

        fit["spec_type"] = "max"
        for idx, col, label in zip([upper_idx, lower_idx], [1, 2], ["Upper", "Lower"]):
            fig, popt, perr = get_hist2d_fit(
                df["SignalParticleK"][idx],
                df["SolarEnergy"][idx],
                fig,
                idx=(1, col),
                per=None,
                acc=75,
                fit=fit,
                zoom=True,
                debug=user_input["debug"],
            )
            popt_corr.append(popt)
            perr_corr.append(perr)

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
                        "ENERGY_AMP": popt_corr[0][0],
                        "INTERSECTION": popt_corr[0][1],
                    },
                    "LOWER": {
                        "OFFSET": popt_int[1],
                        "ENERGY_AMP": popt_corr[1][0],
                        "INTERSECTION": popt_corr[1][1],
                    },
                },
                f,
            )

        rprint(
            f"-> Saved reco energy fit parameters to {root}/config/{config}/{name}/{config}_calib/{config}_{name}_discriminant_calibration.json"
        )
