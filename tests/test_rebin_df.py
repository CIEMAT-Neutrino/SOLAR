import numpy as np
import pandas as pd

from lib.lib_df import rebin_df_columns


def test_rebin_df_columns_two_pass_preserves_counts_and_updates_density():
    df = pd.DataFrame(
        {
            "Energy": [[0.5, 1.5, 2.5, 3.5]],
            "Counts": [[10.0, 20.0, 30.0, 40.0]],
            "Counts/Energy": [[10.0, 20.0, 30.0, 40.0]],
            "Error": [[1.0, 2.0, 3.0, 4.0]],
            "MCCounts": [[100.0, 200.0, 300.0, 400.0]],
        }
    )

    first = rebin_df_columns(df, 2, "Energy", "Counts", "Counts/Energy", "Error", "MCCounts")
    second = rebin_df_columns(first, 2, "Energy", "Counts", "Counts/Energy", "Error", "MCCounts")

    first_counts = np.asarray(first.loc[0, "Counts"], dtype=float)
    first_density = np.asarray(first.loc[0, "Counts/Energy"], dtype=float)
    first_error = np.asarray(first.loc[0, "Error"], dtype=float)
    first_mc_counts = np.asarray(first.loc[0, "MCCounts"], dtype=float)

    assert np.allclose(first_counts, np.asarray([30.0, 70.0]))
    assert np.allclose(first_density, np.asarray([15.0, 35.0]))
    assert np.allclose(first_error, np.asarray([np.sqrt(5.0), 5.0]))
    assert np.allclose(first_mc_counts, np.asarray([300.0, 700.0]))

    second_counts = np.asarray(second.loc[0, "Counts"], dtype=float)
    second_density = np.asarray(second.loc[0, "Counts/Energy"], dtype=float)
    second_error = np.asarray(second.loc[0, "Error"], dtype=float)
    second_mc_counts = np.asarray(second.loc[0, "MCCounts"], dtype=float)

    assert np.allclose(second_counts, np.asarray([100.0]))
    assert np.allclose(second_density, np.asarray([25.0]))
    assert np.allclose(second_error, np.asarray([np.sqrt(30.0)]))
    assert np.allclose(second_mc_counts, np.asarray([1000.0]))

    assert np.isclose(np.sum(np.asarray(df.loc[0, "Counts"], dtype=float)), np.sum(second_counts))


def test_rebin_df_columns_edges_handles_empty_bins_safely():
    df = pd.DataFrame(
        {
            "Energy": [[0.5, 1.5]],
            "Counts": [[2.0, 4.0]],
            "Counts/Energy": [[2.0, 4.0]],
            "Error": [[1.0, 2.0]],
        }
    )

    rebinned = rebin_df_columns(df, np.asarray([0.0, 1.0, 2.0, 3.0]), "Energy", "Counts", "Counts/Energy", "Error")

    assert np.allclose(np.asarray(rebinned.loc[0, "Energy"], dtype=float), np.asarray([0.5, 1.5, 2.5]))
    assert np.allclose(np.asarray(rebinned.loc[0, "Counts"], dtype=float), np.asarray([2.0, 4.0, 0.0]))
    assert np.allclose(np.asarray(rebinned.loc[0, "Counts/Energy"], dtype=float), np.asarray([2.0, 4.0, 0.0]))
    assert np.allclose(np.asarray(rebinned.loc[0, "Error"], dtype=float), np.asarray([1.0, 2.0, 0.0]))


def test_rebin_df_columns_hep_analysis_schema_two_stage_rebin():
    # Mirror the dataframe structure saved in 11AnalysisSignal before rebinned outputs are written.
    df = pd.DataFrame(
        {
            "Geometry": ["hd"],
            "Config": ["hd_1x2x6_centralAPA"],
            "Name": ["marley"],
            "Analysis": ["HEP"],
            "Component": ["hep"],
            "Oscillation": ["Osc"],
            "Mean": ["Mean"],
            "Type": ["signal"],
            "MCCounts": [[10.0, 20.0, 30.0, 40.0]],
            "TrueCounts": [[1.0, 2.0, 3.0, 4.0]],
            "Counts": [[5.0, 15.0, 25.0, 35.0]],
            "Energy": [[10.5, 11.5, 12.5, 13.5]],
            "Error": [[1.0, 2.0, 3.0, 4.0]],
            "Color": ["rgb(204,80,62)"],
            "NHits": [5],
            "OpHits": [8],
            "AdjCl": [12],
        }
    )

    # First pass resembles the static HEP rebin stage in 11AnalysisSignal.
    first = rebin_df_columns(
        df,
        np.asarray([10.0, 12.0, 14.0]),
        "Energy",
        "Counts",
        "Counts/Energy",
        "Error",
        "MCCounts",
    )

    # Second pass emulates a later HEP-only regrouping pass.
    second = rebin_df_columns(
        first,
        np.asarray([10.0, 14.0]),
        "Energy",
        "Counts",
        "Counts/Energy",
        "Error",
        "MCCounts",
    )

    # Rebinning must always act on Counts and preserve total counts.
    assert np.isclose(
        np.sum(np.asarray(df.loc[0, "Counts"], dtype=float)),
        np.sum(np.asarray(first.loc[0, "Counts"], dtype=float)),
    )
    assert np.isclose(
        np.sum(np.asarray(df.loc[0, "Counts"], dtype=float)),
        np.sum(np.asarray(second.loc[0, "Counts"], dtype=float)),
    )

    # Counts/Energy must be derived from rebinned Counts and displayed widths.
    first_counts = np.asarray(first.loc[0, "Counts"], dtype=float)
    first_density = np.asarray(first.loc[0, "Counts/Energy"], dtype=float)
    second_counts = np.asarray(second.loc[0, "Counts"], dtype=float)
    second_density = np.asarray(second.loc[0, "Counts/Energy"], dtype=float)
    assert np.allclose(first_counts, np.asarray([20.0, 60.0]))
    assert np.allclose(first_density, np.asarray([10.0, 30.0]))
    assert np.allclose(second_counts, np.asarray([80.0]))
    assert np.allclose(second_density, np.asarray([20.0]))

    # Non-rebinned metadata should be unchanged.
    for column in [
        "Geometry",
        "Config",
        "Name",
        "Analysis",
        "Component",
        "Oscillation",
        "Mean",
        "Type",
        "Color",
        "NHits",
        "OpHits",
        "AdjCl",
    ]:
        assert second.loc[0, column] == df.loc[0, column]
