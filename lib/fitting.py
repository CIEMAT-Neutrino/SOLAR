from src.utils import get_project_root

from .defaults import get_default_energies, get_default_nhits

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Optional

from rich import print as rprint
from scipy.optimize import curve_fit

np.seterr(divide="ignore", invalid="ignore")

root = get_project_root()
energy_edges, energy_centers, ebin = get_default_energies(root)
nhits = get_default_nhits(root)


def calibration_func(x, a, b, c, d):
    return a * np.exp(-b * x) + c / (1 + np.exp(-d * x))


def resolution(x, p0, p1, p2, b):
    """
    Resolution function.
    """
    residuals = np.sqrt(
        np.power(p2, 2) + np.power(p1 / np.sqrt(x - b), 2) + np.power(p0 / (x - b), 2)
    )
    residuals[np.isnan(residuals)] = 0
    residuals[np.isinf(residuals)] = 0
    return residuals


def peak(x, coefficients, debug=False):
    """
    Peak finder function.
    """
    # Use the scypy find_peaks function to find the peaks
    from scipy.signal import find_peaks

    height = coefficients[0]
    threshold = coefficients[1]
    distance = coefficients[2]
    width = coefficients[3]
    # Find the peaks
    peaks, _ = find_peaks(
        x, height=height, threshold=threshold, distance=distance, width=width
    )
    return peaks


def exp(x, coefficients, debug=False):
    """
    Exponential decay function.
    """
    a = coefficients[0]
    tau = coefficients[1]
    return a * np.exp(-x / tau)


def exp_offset(x, coefficients, debug=False):
    """
    Exponential decay function.
    """
    a = coefficients[0]
    tau = coefficients[1]
    n = coefficients[2]
    return a * np.exp(-x / tau) + n


def gauss(x, coefficients, debug=False):
    """
    Gaussian function.
    """
    a = coefficients[0]
    x0 = coefficients[1]
    sigma = coefficients[2]
    # return a/(sigma*math.sqrt(2*math.pi))*np.exp(-0.5*np.power((x-x0)/sigma,2))
    return a * np.exp(-0.5 * np.power((x - x0) / sigma, 2))


def quadratic(x, coefficients, debug=False):
    """
    Quadratic function.
    """
    a = coefficients[0]
    n = coefficients[1]
    return a * np.power(x, 2) + n


def slope1(x, coefficients, debug=False):
    """
    Linear function.
    """
    m = 1
    n = coefficients[0]
    return m * np.asarray(x) + n


def linear(x, coefficients, debug=False):
    """
    Linear function.
    """
    m = coefficients[0]
    n = coefficients[1]
    return m * np.asarray(x) + n


def quadratic(x, coefficients, debug=False):
    """
    Quadratic function.
    """
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return a * np.power(x, 2) + b * x + n


def fit_hist2d(x, y, z, fit={"func": "exponential"}, debug=False):
    """
    Given a 2D histogram, fit a function to the histogram's cresst.
    """
    if x.shape != z.shape:
        print("\nFlattening 2D histogram...")
        x, y, z = flatten_hist2d(x, y, z, debug=debug)

    if fit["func"] == "exponential":
        print("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return z + exp(x, coefficients, debug=debug)

        initial_guess = (
            1e2,
            1e4,
        )  # Provide an initial guess for the exponential coefficients

    popt, _ = curve_fit(func, x, y, p0=initial_guess)

    return popt


def flatten_hist2d(x, y, z, debug=False):
    """
    Flatten a 2D histogram into a 1D array and extend the x and y arrays to match the flattened array.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        z (array): 2D histogram array.
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
        z (array): 2D histogram array.
    """
    # Print initial shapes of arrays
    if debug:
        print("Initial arrays (x,y,z):", x.shape, y.shape, z.shape, sep=" ")
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    x = np.repeat(x, z.shape[0])
    y = np.tile(y, z.shape[1])
    # Check if the arrays are the same length
    if len(x) != len(y):
        print("x and y arrays are not the same length!")
        print("x: ", len(x), "\ny: ", len(y))
        raise ValueError

    z = z.flatten()
    # Check if the arrays are the same length
    if len(x) != len(z):
        print("x and z arrays are not the same length!")
        print("x: ", len(x), "\nz: ", len(z))
        raise ValueError

    if debug:
        print("Flattened arrays (x,y,z):", len(x), len(y), len(z), sep=" ")
    return x, y, z


def spectrum_hist2d(x, y, z, fit={"threshold": 0, "spec_type": "max"}, debug=False):
    """
    Given a 2D histogram, return the spectrum of the histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        z (array): 2D histogram array.
        fit (dict): dictionary with the fit parameters.
            spec_type (str): spectrum type (max, mean, top, bottom).
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
    """
    # Check shape of z array
    if len(z.shape) != 2:
        print("z array is not 2D!")
        print("z.shape: ", z.shape)
        raise ValueError

    if "threshold" not in fit:
        fit["threshold"] = 0

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    z_max = np.argmax(z, axis=1)
    z_max_value = np.max(z, axis=1)

    if fit["spec_type"] == "max":
        y_max = y[z_max]
        # Filter out the values below the threshold
        x = x[z_max_value > fit["threshold"]]
        y_max = y_max[z_max_value > fit["threshold"]]
        sigma = 1 / np.sqrt(z_max_value[z_max_value > fit["threshold"]])
        return x, y_max, sigma

    if fit["spec_type"] == "mean":
        x_new, y_mean, y_std = [], [], []
        # z is the 2D histogram of data points. Calculate the weighted mean of y for each x using z as weights
        for i, col in enumerate(z):
            if np.sum(col) > fit["threshold"]:
                x_new.append(x[i])
                y_mean.append(np.average(y, weights=col))
                y_std.append(
                    np.sqrt(
                        np.average((y - np.average(y, weights=col)) ** 2, weights=col)
                    )
                )
        y_mean = np.array(y_mean)
        sigma = np.array(y_std) / np.sqrt(
            np.sum(z, axis=1)[np.sum(z, axis=1) > fit["threshold"]]
        )
        return np.asarray(x_new), y_mean, sigma

    if fit["spec_type"] == "top":
        # threshold = 0.25
        z_max = np.max(z, axis=1)
        y_top = np.zeros(len(x))
        sigma_top = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_top[i] = y[j]
                    sigma_top[i] = 1 / np.sqrt(z_max[i])
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_top.shape, sep=" ")
        return x, y_top, sigma_top

    if fit["spec_type"] == "bottom":
        # threshold = 0.20
        z_max = np.max(z, axis=1)
        y_bottom = np.zeros(len(x))
        sigma_bottom = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_bottom[i] = y[j]
                    sigma_bottom[i] = 1 / np.sqrt(z_max[i])
                    break
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_bottom.shape, sep=" ")
        return x, y_bottom, sigma_bottom

    if fit["spec_type"] == "top+bottom":
        z_max = np.max(z, axis=1)
        y_top = np.zeros(len(x))
        sigma_top = np.zeros(len(x))
        y_bottom = np.zeros(len(x))
        sigma_bottom = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_top[i] = y[j]
                    sigma_top[i] = 1 / np.sqrt(z_max[i])

        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_bottom[i] = y[j]
                    sigma_bottom[i] = 1 / np.sqrt(z_max[i])
                    break
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_bottom.shape, sep=" ")
        return x, (y_top, y_bottom), (sigma_top, sigma_bottom)


def plot_hist1d_signal(x, y, signal, fig, idx, debug: bool = False):
    """
    Given an x and y array, plot 2 histograms in the same plot according to the signal index.
    """

    x, y, output = remove_nans_and_infs((x, y), debug=debug)
    signal, output = remove_nans_and_infs(signal, output, debug=debug)

    if len(x) != len(y):
        print("x and y arrays are not the same length!")
        print("x: ", len(x), "\ny: ", len(y))
        raise ValueError

    if len(x) != len(signal):
        print("x and signal arrays are not the same length!")
        print("x: ", len(x), "\nsignal: ", len(signal))
        raise ValueError

    x_array = generate_bins(100, x, debug=debug)
    y_array = generate_bins(100, y, debug=debug)

    if debug and output != None and output != "":
        rprint(output)


def get_hist1d(
    x,
    scan_y=None,
    scan=None,
    per: Optional[tuple] = (1, 99),
    acc=None,
    norm: bool = True,
    density: bool = False,
    debug: bool = False,
):
    """
    Given an x array, generate a 1D histogram.

    Args:
        x (array): x-axis array.
        scan (array): value array.
        per (tuple): percentile range.
        acc (None): define binning according to type in generate_bins.
        norm (bool): If True, the histogram is normalized.
        density (bool): If True, the histogram is normalized.
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
        sigma (array): sigma array.
        labels (str): labels.
        output (str): debug output.
    """
    output = ""
    h, x_bins, sigma, labels = [], [], [], []
    if scan_y is None:
        if len(x) > 0:
            x, this_output = remove_nans_and_infs(x, debug=debug)
            if per is not None:
                x = remove_outliers(x, per=per, debug=debug)
            if isinstance(acc, int):
                x_array = generate_bins(acc, x, debug=debug)
            else:
                x_array = acc

            h, this_x_bins = np.histogram(x, bins=x_array, density=density)

            if norm:
                h = h / (np.sum(h))
            sigma = 1 / np.sqrt(h)
            output += this_output
            return x_array, h, sigma, "Spectrum", output

        else:
            output += "[red]ERROR: Returning empty array![/red]"
            return None, None, None, None, output

    else:
        # Check if scan_y is the same length as x
        if len(scan_y) != len(x):
            print("x and scan_y arrays are not the same length!")
            print("x: ", len(x), "\nscan_y: ", len(scan_y))
            raise ValueError

        x, scan_y, this_output = remove_nans_and_infs((x, scan_y), debug=debug)
        output += this_output

        if per is not None:
            x, scan_y = remove_outliers((x, scan_y), per=per, debug=debug)

        x_array = generate_bins(acc[0], x, debug=debug)
        y_array = generate_bins(acc[1], scan_y, debug=debug)

        for idx, scan_value in enumerate(scan):
            if idx < len(scan) - 1:
                scan_bin = scan[idx + 1] - scan[idx]
            else:
                scan_bin = scan[idx] - scan[idx - 1]

            scan_filter = np.where(
                (x >= (scan_value - scan_bin / 2)) & (x < (scan_value + scan_bin / 2))
            )

            if scan_y is not None:
                this_filtered_y = scan_y[scan_filter]
                try:
                    this_h, this_x_bins = np.histogram(
                        this_filtered_y, bins=y_array, density=density
                    )
                    labels.append(f"{scan_value}")
                    x_array = y_array

                except ValueError:
                    print("y might be empty!")
                    continue
            else:
                this_filtered_x = x[scan_filter]
                this_h, this_x_bins = np.histogram(
                    this_filtered_x, bins=x_array, density=density
                )
                this_sigma = 1 / np.sqrt(this_h)

            if norm:
                this_h = this_h / (np.sum(this_h))
                this_sigma = 1 / np.sqrt(this_h)

            x_bins.append(x_array)
            h.append(this_h)
            sigma.append(this_sigma)

        return x_bins, h, sigma, labels, output


def get_variable_scan(
    x,
    y,
    variable: str = "energy",
    per: tuple = (1, 99),
    norm: bool = True,
    acc=100,
    debug: bool = False,
) -> tuple:
    """
    Given an x array, generate a 1D histogram.

    Args:
        x (array): variable array.
        y (array): value array.
        variable (str): variable to scan (energy, nhits, etc.).
        per (tuple): percentile range.
        norm (bool): If True, the histogram is normalized.
        acc (int)/(tuple): number of bins/(x,y) bins.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
    """
    x, y, output = remove_nans_and_infs((x, y), debug=debug)
    if per is not None:
        x, y = remove_outliers((x, y), per=per, debug=debug)

    mean_variable_array, std_variable_array = [], []
    if variable == "energy":
        for energy in energy_centers:
            energy_filter = np.where(
                (x > (energy - ebin / 2)) & (x < (energy + ebin / 2))
            )
            mean_variable_array.append(np.mean(y[energy_filter]))
            std_variable_array.append(np.std(y[energy_filter]))
        values = energy_centers

    elif variable == "nhits":
        values = []
        for nhit in nhits:
            nhit_filter = np.where(x == nhit)
            if np.sum(nhit_filter) > 0:
                values.append(nhit)
                mean_variable_array.append(np.mean(y[nhit_filter]))
                std_variable_array.append(np.std(y[nhit_filter]))

    else:
        values = generate_bins(acc, x, debug=debug)
        if type(values) is np.ndarray:
            bin_width = values[1] - values[0]
            for value in values:
                value_filter = np.where(
                    (x > (value - bin_width / 2)) & (x < (value + bin_width / 2))
                )
                mean_variable_array.append(np.mean(y[value_filter]))
                std_variable_array.append(np.std(y[value_filter]))
        else:
            if debug:
                rprint("[red]ERROR: Returning empty array![/red]")
            return [0], [0], [0]

    array = np.array(mean_variable_array)
    values, array, output = remove_nans_and_infs((values, array), output, debug=debug)
    array_error = np.array(std_variable_array)

    if norm:
        array = array / np.max(array)
        array_error = array_error / np.max(array)

    if debug:
        rprint(output)
    return values, array, array_error


def fit_hist1d(
    x,
    y,
    sigma: Optional[np.ndarray] = None,
    fit: dict = {
        "bounds": None,
        "func": "linear",
        "trimm": (1, 1),
        "show": True,
        "print": True,
    },
    debug: bool = False,
):
    """
    Given a 1D histogram, fit a function to the histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        func (str): function to fit to histogram (exponential, etc.).
        trimm (tuple[int]): number of bins to remove from the beginning and end of the histogram.
        debug (bool): If True, the debug mode is activated.

    Returns:
        func (function): function to fit to histogram.
    """
    if "trimm" in fit:
        # Check that trimming does not reduce the array to zero length
        if len(x[fit["trimm"][0] : -fit["trimm"][1]]) <= 2:
            rprint("[red]ERROR: Trimming too much![/red]")

        else:
            # Remove x values at the beginning and end of the array
            x = x[fit["trimm"][0] : -fit["trimm"][1]]
            y = y[fit["trimm"][0] : -fit["trimm"][1]]
            sigma = (
                sigma[fit["trimm"][0] : -fit["trimm"][1]] if sigma is not None else None
            )

    # Check if "func" is in the fit dictionary
    if "func" not in fit:
        rprint("[yellow]WARNING: Function not specified![/yellow]")
        return None, None, None, None

    elif fit["func"] == "slope1":
        # if fit["print"] and debug:
        #     rprint("Fitting line...")

        def func(x, *coefficients, debug=False):
            return slope1(x, coefficients, debug=debug)

        labels = ["Intercept"]
        initial_guess = np.random.randn(1)
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([-np.inf], [np.inf])

    elif fit["func"] == "linear":
        # if fit["print"] and debug:
        #     rprint("Fitting line...")

        def func(x, *coefficients, debug=False):
            return linear(x, coefficients, debug=debug)

        labels = ["Slope", "Intercept"]
        initial_guess = [1, 0]
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([0, -np.inf], [0, np.inf])

    elif fit["func"] == "quadratic":
        # if fit["print"] and debug:
        #     rprint("Fitting quadratic...")

        def func(x, *coefficients, debug=False):
            return quadratic(x, coefficients, debug=debug)

        labels = ["Curvature", "Slope", "Intercept"]
        initial_guess = [1e-6, 1, -1]
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([1e-10, 1e-10, -np.inf], [np.inf, np.inf, 5])

    elif fit["func"] == "exponential":
        # if fit["print"] and debug:
        #     rprint("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return exp(x, coefficients, debug=debug)

        labels = ["Amplitude", "Tau"]
        initial_guess = (1e2, 1e4)
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([0, 0], [np.inf, np.inf])

    elif fit["func"] == "exponential_offset":
        # if fit["print"] and debug:
        #     rprint("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return exp(x, coefficients, debug=debug)

        labels = ["Amplitude", "Tau", "Offset"]
        initial_guess = (1e2, 1e4, 0)
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    elif fit["func"] == "gauss":
        # if fit["print"] and debug:
        #     rprint("Fitting gaussian...")

        def func(x, *coefficients, debug=False):
            return gauss(x, coefficients, debug=debug)

        labels = ["Amplitude", "Mean", "Sigma"]
        initial_guess = (np.max(y), x[np.argmax(y)], np.std(y))
        if "bounds" not in fit or fit["bounds"] is None:
            fit["bounds"] = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])
        # initial_guess = (0,0,0)

    else:
        rprint("[red][ERROR][/red]: Function not recognized![/red]")
        return None, None, None, None

    try:
        popt, pcov = curve_fit(
            func,
            x,
            y,
            sigma=sigma,
            p0=initial_guess,
            bounds=fit["bounds"],
            check_finite=True,
        )
        perr = np.sqrt(np.diag(pcov))

    except ValueError:
        rprint("[yellow][WARNING][/yellow] ValueError: `sigma` has incorrect shape.")
        if debug:
            rprint(
                f"[cyan][INFO][/cyan]: x = {np.shape(x)}; y = {np.shape(y)}; sigma = {np.shape(sigma)}"
            )
        popt, pcov = curve_fit(func, x, y, p0=initial_guess, bounds=fit["bounds"])
        perr = np.sqrt(np.diag(pcov))

    return func, labels, popt, perr


def get_hist2d(
    x,
    y,
    per: tuple = (1, 99),
    acc=None,
    norm: bool = False,
    density: bool = False,
    nanz: bool = False,
    logz: bool = False,
    zoom: bool = False,
    debug: bool = False,
):
    """
    Given x and y arrays, generate a 2D histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        acc (int): number of bins.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
        z (array): 2D histogram array.
    """
    # Compute percentile for x & y array determination using a numpy fucntion
    x, y, output = remove_nans_and_infs((x, y), debug=debug)
    if per is not None:
        x, y = remove_outliers((x, y), per=per, debug=debug)

    # Compute the number of bins using the Freedman-Diaconis rule
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    # acc = 2 * IQR(x) / (n^(1/3))
    if acc is None:
        x_array = generate_bins(50, x, debug=debug)
        y_array = generate_bins(50, y, debug=debug)

    elif isinstance(acc, int):
        x_array = generate_bins(acc, x, debug=debug)
        y_array = generate_bins(acc, y, debug=debug)

    elif isinstance(acc, tuple):
        if len(acc) == 1:
            x_array = generate_bins(acc[0], x, debug=debug)
            y_array = generate_bins(acc[0], y, debug=debug)
        elif len(acc) == 2:
            x_array = generate_bins(acc[0], x, debug=debug)
            y_array = generate_bins(acc[1], y, debug=debug)
        else:
            rprint(
                "[red]ERROR[/red]: acc must be a string ('x' or 'y'), an int or a tuple of length 1 or 2!"
            )
            return None, None, None

    elif isinstance(acc, str):
        if acc == "y":
            y_array = generate_bins(50, y, debug=debug)
            x_array = y_array
        elif acc == "x":
            x_array = generate_bins(50, x, debug=debug)
            y_array = x_array

    else:
        rprint(
            "[red]ERROR[/red]: acc must be a string ('x' or 'y'), an int or a tuple of length 1 or 2!"
        )
        return None, None, None

    try:
        h, x_edges, y_edges = np.histogram2d(
            x, y, bins=[x_array, y_array], density=density
        )

    except TypeError:
        h, x_edges, y_edges = np.histogram2d(x, y, density=density)

    if logz:
        h = np.log10(h)

    if norm:
        h = h / (np.sum(h))
    # x, y = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
    x_centers, y_centers = (x_edges[1:] + x_edges[:-1]) / 2, (
        y_edges[1:] + y_edges[:-1]
    ) / 2

    if zoom:
        last_non_zero_x = np.where(np.any(h != 0, axis=1))[0][-1]
        last_non_zero_y = np.where(np.any(h != 0, axis=0))[0][-1]
        x_centers = x_centers[: last_non_zero_x + 1]
        y_centers = y_centers[: last_non_zero_y + 1]
        h = h[: last_non_zero_x + 1, : last_non_zero_y + 1]

    if nanz:
        h = np.where(h == 0, np.nan, h)

    if debug and output != None and output != "":
        rprint(output)

    return x_centers, y_centers, h


def get_hist2d_fit(
    x,
    y,
    fig: go.Figure,
    idx: tuple,
    per: tuple = (1, 99),
    acc: float = 50,
    fit: dict = {
        "color": "grey",
        "opacity": 1,
        "trimm": (1, 1),
        "spec_type": "max",
        "func": "linear",
        "threshold": 0.4,
        "range": (0, 10),
        "show": True,
    },
    density=None,
    nanz: bool = False,
    logz: bool = False,
    zoom: bool = False,
    debug: bool = False,
):
    """
    Given x and y arrays, generate a 2D histogram and fit a function to the histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        fig (plotly figure): plotly figure.
        idx (tuple(int)): (row, col) index of subplot.
        per (tuple): percentile range.
        acc (int): number of bins.
        fit (dict): dictionary with the fit parameters.
        density (bool): If True, the histogram is normalized.
        zoom (bool): If True, the histogram is zoomed in.
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly figure): plotly figure.
        popt (array): array with the fit parameters.
        perr (array): array with the fit errors.
    """
    hx, hy, hz = get_hist2d(
        x, y, per=per, acc=acc, density=density, logz=logz, debug=debug
    )
    plot_z = hz.T.copy()
    if nanz:
        plot_z = np.where(plot_z == 0, np.nan, plot_z)

    fig.add_trace(
        go.Heatmap(z=plot_z, x=hx, y=hy, coloraxis="coloraxis"), row=idx[0], col=idx[1]
    )

    if fit["spec_type"] == "intercept":
        if "range" not in fit:
            fit["range"] = (0, 10)
        popt, perr, labels = [], [], []

        intercepts = find_hist2d_intercept(
            x,
            y,
            acc,
            irange=fit["range"],
            threshold=fit["threshold"],
            show=False,
            debug=debug,
        )
        array = np.arange(np.min(hx), np.max(hx))
        for b in intercepts:
            fig.add_trace(
                go.Scatter(
                    x=array,
                    y=array - b,
                    mode="lines",
                    marker=dict(color=fit["color"], opacity=fit["opacity"]),
                ),
                row=idx[0],
                col=idx[1],
            )
            popt = np.concatenate((popt, [-b]))
            perr = np.concatenate((perr, [10 / acc]))
            labels = np.concatenate((labels, ["Intercept"]))

    else:
        x_spec, y_spec, sigma = spectrum_hist2d(hx, hy, hz, fit=fit, debug=debug)
        if len(x_spec) >= 2:
            func, labels, popt, perr = fit_hist1d(
                x_spec, y_spec, sigma, fit=fit, debug=debug
            )
        else:
            rprint("[red]ERROR[/red]: Not enough data points to fit!")
            return fig, None, None

        fig = plot_hist2d_fit(
            fig=fig,
            idx=idx,
            func=func,
            popt=popt,
            perr=perr,
            x=x_spec,
            y=y_spec,
            fit=fit,
            debug=debug,
        )

    if zoom:
        fig.update_xaxes(range=[np.min(hx), np.max(hx)], row=idx[0], col=idx[1])
        fig.update_yaxes(range=[np.min(hy), np.max(hy)], row=idx[0], col=idx[1])

    if fit["print"]:
        debug_text = [
            "\nFit %s: %.2f +/- %.2f" % (labels[i], popt[i], perr[i])
            for i in range(len(labels))
        ]
        rprint(f"[cyan]INFO: {''.join(debug_text)}[/cyan]")
    return fig, popt, perr


def plot_hist2d_fit(
    fig,
    idx,
    func,
    popt,
    perr,
    x,
    y,
    fit={"color": "grey", "opacity": 1, "show": False},
    debug=False,
):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=func(x, *popt),
            mode="lines",
            marker=dict(color=fit["color"], opacity=fit["opacity"]),
            name="Fit",
        ),
        row=idx[0],
        col=idx[1],
    )
    if fit["show"]:
        if "trimm" in fit:
            x = x[fit["trimm"][0] : -fit["trimm"][1]]
            y = y[fit["trimm"][0] : -fit["trimm"][1]]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(color=fit["color"], opacity=fit["opacity"]),
                name="Spectrum",
                error_y=dict(type="data", array=func(x, *perr), visible=True),
            ),
            row=idx[0],
            col=idx[1],
        )
    return fig


def get_hist1d_fit(
    x,
    fig,
    idx,
    per: Optional[tuple] = (1, 99),
    acc=50,
    fit={"color": "grey", "trimm": (1, 1), "func": "gauss"},
    debug=False,
):
    """
    Given an x array, generate a 1D histogram and fit a function to the histogram.

    Args:
        x (array): x-axis array.
        acc (int): number of bins.
        fig (plotly figure): plotly figure.
        idx (tuple(int)): (row, col) index of subplot.
            row (int): row of subplot.
            col (int): column of subplot.
        func (str): function to fit to histogram (exponential, etc.).
        trimm (tuple[int]): number of bins to remove from the beginning and end of the histogram.
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly figure): plotly figure.
        popt (array): array with the fit parameters.
        perr (array): array with the fit errors.
    """
    x_list, h_list, sigma_list, labels, output = get_hist1d(
        x, per=per, acc=acc, debug=debug
    )
    for x, h, sigma in zip(x_list, h_list, sigma_list):
        fig.add_trace(
            go.Bar(x=x, y=h, marker=dict(color="grey"), name="Spectrum"),
            row=idx[0],
            col=idx[1],
        )
        fig.update_layout(bargap=0)

        try:
            func, labels, popt, perr = fit_hist1d(x, h, sigma, fit=fit, debug=debug)
            # Add text to the plot with the fit parameters
            text = ""
            for i in range(len(labels)):
                text += "%s: %.2f +/- %.2f\n" % (labels[i], popt[i], perr[i])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=func(x, *popt),
                    mode="lines+markers",
                    line=dict(color=fit["color"], shape="hvh"),
                    name="Fit",
                    error_y=dict(type="data", array=func(x, *perr), visible=True),
                ),
                row=idx[0],
                col=idx[1],
            )

        except:
            if fit["print"]:
                rprint("[yellow]WARNING: Fit could not be performed![/yellow]")
            return fig, [], []

        if fit["print"]:
            debug_text = [
                "\nFit %s: %.2f +/- %.2f" % (labels[i], popt[i], perr[i])
                for i in range(len(labels))
            ]
            rprint("[cyan]INFO: " + "".join(debug_text) + "[/cyan]")
    return fig, popt, perr


def find_hist2d_intercept(
    x,
    y,
    acc: int,
    irange: tuple = (0, 10),
    threshold: float = 0.6,
    slope: float = 1,
    show=False,
    debug=False,
) -> list:
    """
    Given x and y arrays, find the intercepts of all crests in the heatmap.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        acc (int): number of bins.
        irange (tuple): range of intercepts.
        threshold (float): threshold for intercepts.
        slope (float): slope of the line.
        show (bool): If True, the debug mode is activated.
        debug (bool): If True, the debug mode is activated.

    Returns:
        intercepts (list): list of intercepts.
    """
    intercepts, counts = [], []
    for idx, b in enumerate(np.linspace(irange[0], irange[1], acc)):
        bins, bar = np.histogram((y + b) / (slope * x), bins=acc)
        if 1 - 1 / acc < bar[np.argmax(bins)] < 1 + 1 / acc:
            if len(intercepts) == 0:
                intercepts.append(b)
                counts.append(bins[np.argmax(bins)])
                if show:
                    plt.step(bar[:-1], bins, where="post", label="b = %f" % b)
            else:
                if b - intercepts[-1] < threshold:
                    if bins[np.argmax(bins)] > counts[-1]:
                        intercepts[-1] = b
                        counts[-1] = bins[np.argmax(bins)]
                        if show:
                            plt.step(bar[:-1], bins, where="post", label="b = %f" % b)
                    # else:
                    #     if debug:
                    #         print("Skipping", b, intercepts[-1])
                else:
                    intercepts.append(b)
                    counts.append(bins[np.argmax(bins)])
                    if show:
                        plt.step(bar[:-1], bins, where="post", label="b = %f" % b)

    if show:
        plt.xlabel(r"($E_{e}$+const) / $E_{\nu}$")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    return intercepts


def get_hist1d_diff(x, y, offset: float = 0, norm: bool = True, debug=False) -> tuple:
    """
    Given an x and y array, return weighted histogram of the difference between x and y.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        offset (float): offset for the difference.
        norm (bool): If True, the histogram is normalized.
        debug (bool): If True, the debug mode is activated.

    Returns:
        intercept: array of intercepts.
        counts: array of counts.
    """
    intercept, counts = [], []
    for b in np.arange(0, np.max(x - y), 0.1):
        diagonal_filter = np.where((x > y + b - 0.05) * (x < y + b + 0.05))
        count = len(diagonal_filter[0])
        intercept.append(b - offset)
        counts.append(count)
    intercept = np.array(intercept)
    counts = np.array(counts)
    if norm:
        count = counts / np.sum(counts)

    return intercept, counts


def generate_bins(acc, x, debug=False):
    if type(acc) == tuple:
        acc = acc[0]

    if type(acc) == int:
        try:
            if (
                type(x[0]) == float
                or type(x[0]) == np.float64
                or type(x[0]) == np.float32
                or type(x[0]) == np.float16
            ):
                bin_width = (np.max(x) - np.min(x)) / acc
                x_array = np.arange(np.min(x), np.max(x) + bin_width, bin_width)
            elif (
                type(x[0]) == int
                or type(x[0]) == np.int64
                or type(x[0]) == np.int32
                or type(x[0]) == np.int16
            ):
                x_array = np.arange(np.min(x) - acc / 2, np.max(x) + 1, acc)
            else:
                if debug:
                    rprint(f"[red]ERROR: x type {type(x[0])} not supported![/red]")
                return None

        except ValueError:
            if debug:
                rprint("[red]ERROR: x might be empty![/red]")
            return None
        except IndexError:
            if debug:
                rprint("[red]ERROR: x might be empty![/red]")
            return None

    elif type(acc) == list:
        x_array = np.asarray(acc)

    else:
        if debug:
            rprint(
                f"[yellow]WARNING: No known binning type, returning {type(acc)}: {acc}![/yellow]"
            )
        return acc

    return x_array


def remove_nans_and_infs(data, output: Optional[bool] = None, debug=False) -> tuple:
    """
    Given an x and y array, remove the values that correspond to inf or nan values.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
    """
    if output is None:
        output = ""

    # if debug:
    #     output += f"[cyan]INFO: Removing nans and infs from data of size {len(data)}...[/cyan]"

    if len(data) > 2:
        x = data
        initial_len = len(x)
        x = np.asarray(x)

        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            x = x[np.where(np.isinf(x) == False)]
            x = x[np.where(np.isnan(x) == False)]
            reduced_len = len(x)
            # if debug:
            #     output += f"[yellow]WARNING: x contains inf or nan values! Reduced size from {initial_len} to {reduced_len}[/yellow]"

        return x, output

    elif len(data) == 2 and len(data[0]) == len(data[1]):
        x, y = data
        initial_len = len(x)
        x = np.asarray(x)
        y = np.asarray(y)

        # if debug:
        #     output += f"[cyan]INFO: Removing outliers from x and y arrays...[/cyan]"

        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            x = x[np.where(np.isinf(x) == False)]
            y = y[np.where(np.isinf(x) == False)]
            x = x[np.where(np.isnan(x) == False)]
            y = y[np.where(np.isnan(x) == False)]
            reduced_len = len(x)

            if debug:
                output += f"[yellow]WARNING: x contains inf or nan values! Reduced size from {initial_len} to {reduced_len}[/yellow]"

        if np.any(np.isinf(y)) or np.any(np.isnan(y)):
            y = y[np.where(np.isinf(y) == False)]
            x = x[np.where(np.isinf(y) == False)]
            y = y[np.where(np.isnan(y) == False)]
            x = x[np.where(np.isnan(y) == False)]
            reduced_len = len(y)

            if debug:
                output += f"[yellow]WARNING: y contains inf or nan values! Reduced size from {initial_len} to {reduced_len}[/yellow]"

        return x, y, output

    elif len(data) == 0:
        output += "[yellow]WARNING: Size 0 data![/yellow]"
        return data, output

    else:
        output += "[red]ERROR: Could not remove outliers![/red]"
        return data, output


def remove_outliers(
    data: tuple, per: Optional[tuple] = (1, 99), debug: bool = False
) -> tuple:
    """
    Given an x and y array, remove the values that correspond to outliers.

    Args:
        data (tuple): x and y arrays.
        per (tuple): percentile range.
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
    """
    if per is None:
        return data

    if len(data) > 2:
        x = data
        try:
            lims = np.percentile(x, per, axis=0, keepdims=False)
            reduced_x = [i for i in x if lims[0] <= i <= lims[1]]
            # if debug:
            #     rprint(
            #         f"[cyan]INFO: Percentile limits: {str(lims)} reducing array to {100*len(reduced_x)/len(x):.1f}%[/cyan]"
            #     )
            x = np.asarray(reduced_x)

        except:
            if debug:
                rprint("[red]ERROR: Could not remove outliers![/red]")

        return x

    elif len(data) == 2 and len(data[0]) == len(data[1]):
        x, y = data
        try:
            lims = np.percentile([x, y], per, axis=1, keepdims=False)
            reduced_x = [
                i
                for i, j in zip(x, y)
                if lims[0][0] <= i <= lims[1][0] and lims[0][1] <= j <= lims[1][1]
            ]
            reduced_y = [
                j
                for i, j in zip(x, y)
                if lims[0][0] <= i <= lims[1][0] and lims[0][1] <= j <= lims[1][1]
            ]
            # if debug:
            #     rprint(
            #         f"[cyan]INFO: Percentile limits: {str(lims)} reducing arrays to x:{100*len(reduced_x)/len(x):.1f}% and y:{100*len(reduced_y)/len(y):.1f}%[/cyan]"
            #     )
            x, y = np.asarray(reduced_x), np.asarray(reduced_y)

        except:
            if debug:
                rprint("[red]ERROR: Could not remove outliers![/red]")

        return x, y

    else:
        if debug:
            rprint("[red]ERROR: Could not remove outliers![/red]")
        return data


# ─── Sensitivity chi² parallelisation helpers ────────────────────────────────
# These are module-level functions so ProcessPoolExecutor can pickle them on
# any platform (fork AND spawn).  Keep free of global state.

def _sensitivity_apply_energy_scale(
    template_2d: np.ndarray,
    delta_e: float,
    e_centers: np.ndarray,
) -> np.ndarray:
    """Shift template energy axis by fractional delta_e via linear interpolation."""
    if delta_e == 0.0:
        return template_2d
    query = e_centers / (1.0 + delta_e)
    out = np.empty_like(template_2d)
    for row in range(template_2d.shape[0]):
        out[row] = np.interp(query, e_centers, template_2d[row], left=0.0, right=0.0)
    return out


def _sensitivity_fit_with_escale(
    obs: np.ndarray,
    pred: np.ndarray,
    bkg: np.ndarray,
    e_centers: np.ndarray,
    sigma_pred: float,
    sigma_bkg: float,
    sigma_e: float,
    n_sigma_bound: float = 5.0,
    fit_background: bool = True,
    use_legacy_background_penalty: bool = False,
) -> tuple:
    """Profile chi² over energy-scale nuisance delta_e with Gaussian pull."""
    from scipy.optimize import minimize_scalar
    from lib.root import Sensitivity_Fitter

    def _objective(de):
        shifted = _sensitivity_apply_energy_scale(pred, de, e_centers)
        f = Sensitivity_Fitter(
            obs, shifted, bkg,
            SigmaPred=sigma_pred, SigmaBkg=sigma_bkg,
            bb_mask=(bkg > 0),
            fit_background=fit_background,
            use_legacy_background_penalty=use_legacy_background_penalty,
        )
        c, _, _ = f.Fit(0.0, 0.0, debug=False)
        # Guard against None and invalid values
        if c is None or not np.isfinite(c) or c < 0:
            c = 1e6  # Large but not capped
        return float(c) + (de / sigma_e) ** 2

    result = minimize_scalar(
        _objective,
        bounds=(-n_sigma_bound * sigma_e, n_sigma_bound * sigma_e),
        method="bounded",
    )
    chi2_val = result.fun
    # Guard against invalid results
    if not np.isfinite(chi2_val) or chi2_val < 0:
        chi2_val = 1e6
    return float(chi2_val), None, None


def sensitivity_chi2_worker(task: dict) -> tuple:
    """
    Compute (params, solar_chi2, react_chi2) for one oscillation grid point.

    Top-level callable for ProcessPoolExecutor.map.  All inputs are passed
    explicitly via the task dict — no global state.

    task keys
    ---------
    params               : (dm2, sin13, sin12) tuple
    obs                  : observed template array, shape (N_nadir, N_energy)
    pred1                : solar prediction template (same shape)
    pred2                : reactor prediction template (same shape)
    bkg                  : background template (same shape)
    sigma_pred           : signal systematic uncertainty
    sigma_bkg            : background systematic uncertainty
    marginalize_e_scale  : bool — profile over energy scale
    sigma_e_scale        : energy scale uncertainty (fractional)
    e_centers_thld       : energy bin centers above analysis threshold
    fit_background       : bool — if True, fit background normalization as free parameter
    use_legacy_background_penalty: bool — if True, use constant penalty for background uncertainty
                                         (legacy mode). If False (default), profile over background.
    fit_method           : "legacy" (default, this nested-minimiser path) or "pull"
                           (dispatches to sensitivity_chi2_worker_pull)
    return_diagnostics   : append {"solar": {...}, "react": {...}} with chi2_zero (chi2 at
                           nominal nuisances) so sensitivity_validation_gates can check the fit
    """
    if task.get("fit_method", "legacy") == "pull":
        return sensitivity_chi2_worker_pull(task)

    from lib.root import Sensitivity_Fitter

    params = task["params"]
    obs    = task["obs"]
    pred1  = task["pred1"]
    pred2  = task["pred2"]
    bkg    = task["bkg"]
    sp     = task["sigma_pred"]
    sb     = task["sigma_bkg"]
    use_esc = task["marginalize_e_scale"]
    sig_e   = task["sigma_e_scale"]
    e_ctr   = task["e_centers_thld"]
    fit_bkg = task.get("fit_background", True)
    use_legacy = task.get("use_legacy_background_penalty", False)

    # Solar chi²
    if use_esc:
        solar_chi2, _, _ = _sensitivity_fit_with_escale(
            obs, pred1, bkg, e_ctr, sp, sb, sig_e, 
            fit_background=fit_bkg, 
            use_legacy_background_penalty=use_legacy
        )
    else:
        f = Sensitivity_Fitter(
            obs, pred1, bkg, 
            SigmaPred=sp, SigmaBkg=sb, 
            bb_mask=(bkg > 0), 
            fit_background=fit_bkg,
            use_legacy_background_penalty=use_legacy
        )
        solar_chi2, _, _ = f.Fit(0.0, 0.0, debug=False)
    
    # Validate solar_chi2 - only guard against invalid values, not cap valid ones
    if solar_chi2 is not None:
        if not np.isfinite(solar_chi2) or solar_chi2 < 0:
            solar_chi2 = 1e6  # Replace invalid with large value
    
    # Preserve serial semantics: skip reactor if solar fit failed
    if solar_chi2 is None:
        return (params, None, None, None) if task.get("return_diagnostics") else (params, None, None)

    # Reactor chi²
    if use_esc:
        react_chi2, _, _ = _sensitivity_fit_with_escale(
            obs, pred2, bkg, e_ctr, sp, sb, sig_e,
            fit_background=fit_bkg,
            use_legacy_background_penalty=use_legacy
        )
    else:
        f = Sensitivity_Fitter(
            obs, pred2, bkg,
            SigmaPred=sp, SigmaBkg=sb,
            bb_mask=(bkg > 0),
            fit_background=fit_bkg,
            use_legacy_background_penalty=use_legacy
        )
        react_chi2, _, _ = f.Fit(0.0, 0.0, debug=False)
    
    # Validate react_chi2 - only guard against invalid values
    if react_chi2 is not None:
        if not np.isfinite(react_chi2) or react_chi2 < 0:
            react_chi2 = 1e6  # Replace invalid with large value

    if task.get("return_diagnostics"):
        diagnostics = {}
        for label, pred in (("solar", pred1), ("react", pred2)):
            nominal = Sensitivity_Fitter(
                obs, pred, bkg,
                SigmaPred=sp, SigmaBkg=sb,
                bb_mask=(bkg > 0),
                fit_background=fit_bkg,
                use_legacy_background_penalty=use_legacy
            )
            diagnostics[label] = {"chi2_zero": float(nominal.NumpyOperator(0.0, 0.0))}
        return params, solar_chi2, react_chi2, diagnostics

    return params, solar_chi2, react_chi2


# ─── Pull-method sensitivity chi² (closed-form profiling) ────────────────────
# Alternative to the nested-minimiser path above, selected with --fit_method pull.
#
# Nuisances enter the expectation linearly, mu(alpha) = mu0 + J^T alpha. That is EXACT for
# the signal and background normalisations and a first-order (secant over +-1 sigma)
# approximation for the energy scale and sin^2(theta13). The Gaussian-approximation profile
# then has a closed form (the "pull method", Fogli et al., hep-ph/0206162), and the Poisson
# profile is reached from it with a few exact-Hessian Newton steps: the deviance is convex
# in mu and mu is linear in alpha, so the problem is convex and Newton converges
# quadratically. There are no bounded 1D searches, no finite-difference gradients and no
# failure sentinels, so chi2 is a smooth function of the templates at any background level
# (the legacy path breaks down once N_bkg * sigma_bkg^2 >> 1, e.g. HD with ~1e13 events).

SENSITIVITY_FIT_METHODS = ("legacy", "pull")
SENSITIVITY_CHI2_SENTINEL = 1e6   # value the legacy path writes when a fit fails


def _poisson_deviance_terms(obs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-bin 2[mu - o + o ln(o/mu)], numerically stable when o ~ mu >> 1."""
    out = 2.0 * mu   # o == 0 limit
    pos = obs > 0
    x = (obs[pos] - mu[pos]) / mu[pos]
    f = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    # (1+x) ln(1+x) - x as a series: the closed form cancels catastrophically for |x| << 1
    f[small] = xs * xs * (0.5 - xs * (1.0 / 6.0 - xs * (1.0 / 12.0 - xs / 20.0)))
    xl = x[~small]
    f[~small] = (1.0 + xl) * np.log1p(xl) - xl
    out[pos] = 2.0 * mu[pos] * f
    return out


def _solve_equilibrated(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a small SPD system after Jacobi scaling (diagonals can span ~10 decades)."""
    d = 1.0 / np.sqrt(np.abs(np.diag(matrix)))
    scaled = matrix * d[:, None] * d[None, :]
    try:
        y = np.linalg.solve(scaled, rhs * d)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(scaled, rhs * d, rcond=None)[0]
    return y * d


def sensitivity_pull_jacobian(
    pred: np.ndarray,
    bkg: np.ndarray,
    e_centers: Optional[np.ndarray] = None,
    sigma_pred: float = 0.0,
    sigma_bkg: float = 0.0,
    sigma_e_scale: float = 0.0,
    dpred_dsin13: Optional[np.ndarray] = None,
    sigma_sin13: float = 0.0,
) -> tuple:
    """Nuisance response templates d(mu)/d(alpha_k), their prior widths and names.

    Nuisances with a zero (or missing) prior width are fixed and left out.
    """
    pred = np.asarray(pred, dtype=float)
    cols, sigmas, names = [], [], []
    if sigma_pred and sigma_pred > 0:
        cols.append(pred)
        sigmas.append(float(sigma_pred))
        names.append("signal_norm")
    if sigma_bkg and sigma_bkg > 0:
        cols.append(np.asarray(bkg, dtype=float))
        sigmas.append(float(sigma_bkg))
        names.append("background_norm")
    if sigma_e_scale and sigma_e_scale > 0:
        if e_centers is None:
            raise ValueError("sensitivity_pull_jacobian: energy-scale nuisance needs e_centers")
        up = _sensitivity_apply_energy_scale(pred, +float(sigma_e_scale), e_centers)
        down = _sensitivity_apply_energy_scale(pred, -float(sigma_e_scale), e_centers)
        cols.append((up - down) / (2.0 * float(sigma_e_scale)))
        sigmas.append(float(sigma_e_scale))
        names.append("energy_scale")
    if dpred_dsin13 is not None and sigma_sin13 and sigma_sin13 > 0:
        cols.append(np.asarray(dpred_dsin13, dtype=float))
        sigmas.append(float(sigma_sin13))
        names.append("sin13")
    jac = np.stack(cols) if cols else np.zeros((0,) + pred.shape)
    return jac, np.asarray(sigmas, dtype=float), names


def sensitivity_pull_profile(
    obs: np.ndarray,
    mu0: np.ndarray,
    jac: np.ndarray,
    sigma: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> dict:
    """Profile chi2 = min_alpha D_Poisson(obs, mu0 + J^T alpha) + sum_k (alpha_k/sigma_k)^2.

    Returns a dict with chi2 (Poisson profile), chi2_gauss (closed-form Pearson profile),
    chi2_zero (all nuisances at nominal, an upper bound on chi2), alpha, n_iter, converged.
    """
    obs = np.asarray(obs, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    sel = mu0 > 0
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    o, m0 = obs[sel], mu0[sel]
    sigma = np.asarray(sigma, dtype=float)
    k = sigma.size

    chi2_zero = float(_poisson_deviance_terms(o, m0).sum())
    result = {
        "chi2": chi2_zero, "chi2_gauss": chi2_zero, "chi2_zero": chi2_zero,
        "alpha": np.zeros(k), "n_iter": 0, "converged": True, "n_bins": int(o.size),
    }
    if k == 0 or o.size == 0:
        return result

    # Work in standardised nuisances beta = alpha / sigma (unit Gaussian priors).
    a = np.asarray(jac, dtype=float)[:, sel] * sigma[:, None]
    eye = np.eye(k)

    def _objective(beta):
        mu = m0 + a.T @ beta
        if np.any(mu <= 0):
            return np.inf, mu
        return float(_poisson_deviance_terms(o, mu).sum() + beta @ beta), mu

    # Closed-form Gaussian profile, V = diag(mu0): chi2 = r^T (V + A^T A)^-1 r (Woodbury).
    r = o - m0
    aw = a / m0
    b = aw @ r
    beta = _solve_equilibrated(eye + aw @ a.T, b)
    result["chi2_gauss"] = float(r @ (r / m0) - b @ beta)

    f_best, mu = _objective(beta)
    if not np.isfinite(f_best) or f_best > chi2_zero:
        beta, f_best, mu = np.zeros(k), chi2_zero, m0.copy()

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        ratio = o / mu
        grad = 2.0 * (a @ (1.0 - ratio)) + 2.0 * beta
        hess = 2.0 * ((a * (ratio / mu)) @ a.T) + 2.0 * eye
        step = -_solve_equilibrated(hess, grad)
        decrement = -float(grad @ step)          # Newton decrement^2 (~2x expected gain)
        if decrement <= 2.0 * tol:
            converged = True
            break
        t = 1.0
        while t > 1e-12:
            f_new, mu_new = _objective(beta + t * step)
            if f_new <= f_best - 1e-4 * t * decrement:
                break
            t *= 0.5
        else:
            # No sufficient decrease: we are at the floating-point floor of the objective.
            converged = decrement <= 1e-6 * max(1.0, f_best)
            break
        beta = beta + t * step
        f_best, mu = f_new, mu_new

    result.update(
        chi2=float(f_best), alpha=beta * sigma, n_iter=int(n_iter), converged=bool(converged)
    )
    return result


def sensitivity_pull_chi2(
    obs: np.ndarray,
    pred: np.ndarray,
    bkg: np.ndarray,
    sigma_pred: float = 0.0,
    sigma_bkg: float = 0.0,
    e_centers: Optional[np.ndarray] = None,
    sigma_e_scale: float = 0.0,
    dpred_dsin13: Optional[np.ndarray] = None,
    sigma_sin13: float = 0.0,
) -> dict:
    """One-call pull-method fit of pred + bkg to obs (bins with bkg > 0, as in the legacy fitter)."""
    bkg = np.asarray(bkg, dtype=float)
    pred = np.asarray(pred, dtype=float)
    jac, sigma, names = sensitivity_pull_jacobian(
        pred, bkg, e_centers, sigma_pred, sigma_bkg, sigma_e_scale, dpred_dsin13, sigma_sin13
    )
    res = sensitivity_pull_profile(
        obs, pred + bkg, jac, sigma, mask=(bkg > 0) if np.any(bkg > 0) else None
    )
    res["nuisances"] = dict(zip(names, (float(x) for x in res["alpha"])))
    return res


def sensitivity_chi2_worker_pull(task: dict) -> tuple:
    """Pull-method counterpart of sensitivity_chi2_worker (same task keys).

    Extra optional task keys
    ------------------------
    dpred1_dsin13, dpred2_dsin13 : d(pred)/d(sin^2 theta13) templates; sin13 is profiled
                                   inside the fit when these and sigma_sin13 > 0 are given
    sigma_sin13                  : prior width on sin^2 theta13
    pull_jacobians               : {"solar": (jac, sigma, names), "react": (...)} precomputed
                                   once per scan so workers skip the energy-scale shifts
    return_diagnostics           : append a per-fit diagnostics dict to the returned tuple
    """
    params = task["params"]
    obs = np.asarray(task["obs"], dtype=float)
    bkg = np.asarray(task["bkg"], dtype=float)
    mask = (bkg > 0) if np.any(bkg > 0) else None
    precomputed = task.get("pull_jacobians") or {}

    chi2, diagnostics = {}, {}
    for label, pred_key, dsin13_key in (
        ("solar", "pred1", "dpred1_dsin13"),
        ("react", "pred2", "dpred2_dsin13"),
    ):
        pred = np.asarray(task[pred_key], dtype=float)
        if label in precomputed:
            jac, sigma, names = precomputed[label]
        else:
            jac, sigma, names = sensitivity_pull_jacobian(
                pred, bkg,
                e_centers=task.get("e_centers_thld"),
                sigma_pred=task.get("sigma_pred", 0.0),
                sigma_bkg=task.get("sigma_bkg", 0.0),
                sigma_e_scale=task.get("sigma_e_scale", 0.0) if task.get("marginalize_e_scale") else 0.0,
                dpred_dsin13=task.get(dsin13_key),
                sigma_sin13=task.get("sigma_sin13", 0.0),
            )
        res = sensitivity_pull_profile(obs, pred + bkg, jac, sigma, mask=mask)
        chi2[label] = res["chi2"]
        diagnostics[label] = {
            "chi2_zero": res["chi2_zero"],
            "chi2_gauss": res["chi2_gauss"],
            "converged": res["converged"],
            "n_iter": res["n_iter"],
            "nuisances": dict(zip(names, (float(x) for x in res["alpha"]))),
        }

    out = (params, chi2["solar"], chi2["react"])
    return out + (diagnostics,) if task.get("return_diagnostics") else out


def _grid_spikes(grid, abs_tol: float, rel_tol: float, level_tol: float = 0.0) -> list:
    """Isolated extrema along either axis of a 2D chi2 grid.

    A cell is a spike when it is a strict local extremum along an axis and BOTH jumps to its
    neighbours exceed max(abs_tol, rel_tol * the neighbours' own outward steps,
    level_tol * its Delta chi2 above the grid minimum). The minimum of a smooth bowl has jumps
    ~c*h^2 against outward steps ~3*c*h^2, so it is not flagged; oscillating fit noise and
    isolated outliers are. The level term ignores percent-level template roughness far outside
    any contour of interest (e.g. a 1.4 bump at Delta chi2 = 74).
    """
    values = np.asarray(grid.to_numpy(dtype=float))
    levels_all = values - np.nanmin(values)
    spikes = []
    for axis in (0, 1):
        v = values if axis == 0 else values.T
        lv = levels_all if axis == 0 else levels_all.T
        n = v.shape[0]
        for i in range(1, n - 1):
            left, centre, right = v[i - 1], v[i], v[i + 1]
            d_left, d_right = centre - left, centre - right
            extremum = (np.sign(d_left) == np.sign(d_right)) & (d_left != 0)
            outer = []
            if i >= 2:
                outer.append(np.abs(v[i - 1] - v[i - 2]))
            if i + 2 < n:
                outer.append(np.abs(v[i + 2] - v[i + 1]))
            outer_step = np.nanmean(np.stack(outer), axis=0) if outer else np.zeros_like(centre)
            jump = np.minimum(np.abs(d_left), np.abs(d_right))
            threshold = np.maximum(np.maximum(abs_tol, rel_tol * outer_step), level_tol * lv[i])
            flagged = extremum & (jump > threshold)
            for j in np.flatnonzero(flagged):
                row, col = (i, j) if axis == 0 else (j, i)
                spikes.append({
                    "dm2": float(grid.index[row]), "value": float(grid.columns[col]),
                    "chi2": float(values[row, col]), "jump": float(jump[j]), "axis": "dm2" if axis == 0 else "x",
                })
    unique = {(s["dm2"], s["value"]): s for s in sorted(spikes, key=lambda s: s["jump"])}
    return sorted(unique.values(), key=lambda s: -s["jump"])


def sensitivity_validation_gates(
    solar_df,
    react_df,
    grids: dict,
    diagnostics: Optional[dict] = None,
    solar_point: Optional[tuple] = None,
    react_point: Optional[tuple] = None,
    sin13_profile: str = "none",
    asimov_tol: float = 0.01,
    bound_tol: float = 1e-3,
    spike_abs_tol: float = 1.0,
    spike_rel_tol: float = 1.0,
    spike_level_tol: float = 0.05,
    max_report: int = 10,
) -> dict:
    """Method-independent sanity checks on a sensitivity chi2 scan.

    Gates
    -----
    finite         every chi2 is finite and none equals the legacy failure sentinel
    asimov         Asimov data at the reference point gives chi2 ~ 0 for its own fit
    profile_bound  profiled chi2 <= chi2 with every nuisance at nominal (needs diagnostics)
    smoothness     no isolated spikes in the 2D grids (see _grid_spikes)
    sin13_profile  grid-minimum sin13 profiling only when every (dm2, sin12) cell has
                   more than one sin13 value; otherwise it paints a cross artifact

    Returns {"passed": bool, "gates": {name: {...}}}. A gate that cannot be evaluated
    reports passed=None and does not fail the scan.
    """
    gates = {}

    def _chi2_values(df):
        return np.asarray(df["chi2"], dtype=float) if len(df) else np.zeros(0)

    solar_vals, react_vals = _chi2_values(solar_df), _chi2_values(react_df)
    n_nonfinite = int((~np.isfinite(solar_vals)).sum() + (~np.isfinite(react_vals)).sum())
    n_sentinel = int((solar_vals >= SENSITIVITY_CHI2_SENTINEL).sum() + (react_vals >= SENSITIVITY_CHI2_SENTINEL).sum())
    gates["finite"] = {
        "passed": n_nonfinite == 0 and n_sentinel == 0,
        "n_nonfinite": n_nonfinite, "n_sentinel": n_sentinel,
        "n_points": int(solar_vals.size),
    }

    def _chi2_at(df, point):
        if point is None or not len(df):
            return None
        hit = df[
            np.isclose(df["dm2"].astype(float), point[0])
            & np.isclose(df["sin13"].astype(float), point[1])
            & np.isclose(df["sin12"].astype(float), point[2])
        ]
        return float(hit["chi2"].astype(float).iloc[0]) if len(hit) else None

    solar_at_solar, react_at_react = _chi2_at(solar_df, solar_point), _chi2_at(react_df, react_point)
    evaluated = [v for v in (solar_at_solar, react_at_react) if v is not None]
    gates["asimov"] = {
        "passed": (all(v <= asimov_tol for v in evaluated) if evaluated else None),
        "solar_fit_at_solar": solar_at_solar, "react_fit_at_react": react_at_react,
        "tolerance": asimov_tol,
    }

    if diagnostics:
        violations = []
        for fit_label, df in (("solar", solar_df), ("react", react_df)):
            for dm2_v, sin13_v, sin12_v, chi2_v in df[["dm2", "sin13", "sin12", "chi2"]].astype(float).itertuples(index=False):
                diag = (diagnostics.get((dm2_v, sin13_v, sin12_v)) or {}).get(fit_label)
                if not diag or diag.get("chi2_zero") is None:
                    continue
                bound = float(diag["chi2_zero"])
                excess = chi2_v - bound
                if excess > bound_tol + 1e-9 * abs(bound):
                    violations.append({"fit": fit_label, "dm2": dm2_v, "sin13": sin13_v, "sin12": sin12_v,
                                       "chi2": chi2_v, "chi2_zero": bound, "excess": excess})
        violations.sort(key=lambda v: -v["excess"])
        not_converged = sum(
            1 for d in diagnostics.values() for fit in (d or {}).values()
            if isinstance(fit, dict) and fit.get("converged") is False
        )
        gates["profile_bound"] = {
            "passed": not violations, "n_violations": len(violations),
            "n_not_converged": int(not_converged), "worst": violations[:max_report],
        }
    else:
        gates["profile_bound"] = {"passed": None, "reason": "no per-point diagnostics"}

    smooth = {}
    for grid_name, grid in grids.items():
        spikes = _grid_spikes(grid.astype(float), spike_abs_tol, spike_rel_tol, spike_level_tol)
        smooth[grid_name] = {"n_spikes": len(spikes), "worst": spikes[:max_report]}
    gates["smoothness"] = {
        "passed": all(v["n_spikes"] == 0 for v in smooth.values()),
        "abs_tol": spike_abs_tol, "rel_tol": spike_rel_tol, "level_tol": spike_level_tol, "grids": smooth,
    }

    if sin13_profile == "grid" and len(solar_df):
        counts = solar_df.groupby(["dm2", "sin12"]).size()
        coverage = float((counts > 1).mean())
        gates["sin13_profile"] = {
            "passed": coverage == 1.0, "mode": "grid",
            "fraction_cells_with_multiple_sin13": coverage,
        }
    else:
        gates["sin13_profile"] = {"passed": None, "mode": sin13_profile}

    return {
        "passed": all(g["passed"] is not False for g in gates.values()),
        "gates": gates,
    }
