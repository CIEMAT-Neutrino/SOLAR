from src.utils import get_project_root

from .ana_functions import get_default_energies, get_default_nhits

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from rich import print as rprint
from scipy.optimize import curve_fit
np.seterr(divide='ignore', invalid='ignore')

root = get_project_root()
energy_edges, energy_centers, ebin = get_default_energies(root)
nhits = get_default_nhits(root)

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
    peaks, _ = find_peaks(x, height=height, threshold=threshold, distance=distance, width=width)
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


def polynomial(x, coefficients, debug=False):
    """
    Polynomial function.
    """
    if debug:
        print("Polynomial coefficients: ", coefficients)
    return np.polyval(coefficients, x)


def fit_hist2d(x, y, z, fit={"func": "polynomial"}, debug=False):
    """
    Given a 2D histogram, fit a function to the histogram's cresst.
    """
    if x.shape != z.shape:
        print("\nFlattening 2D histogram...")
        x, y, z = flatten_hist2d(x, y, z, debug=debug)

    if fit["func"] == "polynomial":
        print("Fitting polynomial...")

        def func(x, *coefficients, debug=False):
            return z + polynomial(x, coefficients, debug=debug)

        initial_guess = np.random.randn(
            3
        )  # Provide an initial guess for the polynomial coefficients

    if fit["func"] == "exponential":
        print("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return z + exp(x, coefficients, debug=debug)

        initial_guess = (
            1e2,
            1e4,
        )  # Provide an initial guess for the exponential coefficients

    # Fitting the polynomial line to the data
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

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    z_max = np.argmax(z, axis=1)

    if fit["spec_type"] == "max":
        y_max = y[z_max]
        return x, y_max

    if fit["spec_type"] == "mean":
        y_mean = np.sum(y * z, axis=1) / np.sum(z, axis=1)
        return x, y_mean

    if fit["spec_type"] == "top":
        # threshold = 0.25
        z_max = np.max(z, axis=1)
        y_top = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_top[i] = y[j]
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_top.shape, sep=" ")
        return x, y_top

    if fit["spec_type"] == "bottom":
        # threshold = 0.20
        z_max = np.max(z, axis=1)
        y_bottom = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_bottom[i] = y[j]
                    break
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_bottom.shape, sep=" ")
        return x, y_bottom

    if fit["spec_type"] == "top+bottom":
        # threshold = 0.35
        z_max = np.max(z, axis=1)
        y_top = np.zeros(len(x))
        y_bottom = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_top[i] = y[j]
        # threshold = 0.35
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i, j] > z_max[i] * fit["threshold"]:
                    y_bottom[i] = y[j]
                    break
        if debug:
            print("Spectrum arrays (x,y):", x.shape, y_bottom.shape, sep=" ")
        return x, y_top, y_bottom


def generate_bins(acc, x, debug=False):
    if type(acc) == int:
        x_array = np.linspace(np.min(x), np.max(x), acc + 1)
    elif type(acc) == tuple:
        x_array = np.linspace(np.min(x), np.max(x), acc[0] + 1)
    elif type(acc) == list or type(acc) == np.ndarray:
        x_array = acc
    elif acc == None:
        x_array = "auto"
    elif type(acc) == str:
        pass
    else:
        rprint("[red]ERROR: acc must be an integer, list, or numpy array![/red]")
        raise ValueError
    
    return x_array

def get_hist1d(x, per:tuple = (1, 99), acc = None, norm = True, density = False, debug = False):
    """
    Given an x array, generate a 1D histogram.

    Args:
        x (array): x-axis array.
        per (tuple): percentile range.
        acc (None): define binning according to type in generate_bins.
        norm (bool): If True, the histogram is normalized.
        density (bool): If True, the histogram is normalized.
        debug (bool): If True, the debug mode is activated.

    Returns:
        x (array): x-axis array.
        y (array): y-axis array.
    """
    x = np.asarray(x)
    try:
        lims = np.percentile(x, per, axis=0, keepdims=False)
        x = [i for i in x if lims[0] < i < lims[1]]
    except IndexError:
        pass

    if debug:
        rprint(type(x), x, "\n[cyan]INFO: Percentile limits: " + str(lims) + "[/cyan]")

    x_array = generate_bins(acc, x, debug=debug)
    h, x = np.histogram(x, bins=x_array, density=density)

    if norm:
        h = h / (np.sum(h))
    x = (x[1:] + x[:-1]) / 2

    return x, h


def get_variable_scann(x, y, variable="energy", per:tuple = (1, 99), norm = True, acc = 100, debug = False):
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
    mean_variable_array = []
    std_variable_array = []
    try:
        lims = np.percentile(y, per, axis=0, keepdims=False)
        x = [x[i] for i in range(len(x)) if lims[0] < y[i] < lims[1]]
        y = [i for i in y if lims[0] < i < lims[1]]
    except IndexError:
        pass

    x = np.asarray(x)
    y = np.asarray(y)
    
    if debug:
        rprint(type(x), x, "\n[cyan]INFO: Percentile limits: " + str(lims) + "[/cyan]")

    if variable == "energy":
        for energy in energy_centers:
            energy_filter = np.where((x > (energy-ebin/2)) & (x < (energy+ebin/2)))
            mean_variable_array.append(np.mean(y[energy_filter]))
            std_variable_array.append(np.std(y[energy_filter]))
        values = energy_centers

    elif variable == "nhits":
        for nhit in nhits:
            nhit_filter = np.where(x == nhit)
            mean_variable_array.append(np.mean(y[nhit_filter]))
            std_variable_array.append(np.std(y[nhit_filter]))
        values = nhits

    else:
        values = generate_bins(acc, x, debug=debug)
        bin_width = values[1] - values[0]
        for value in values:
            value_filter = np.where((x > (value-bin_width/2)) & (x < (value+bin_width/2)))
            mean_variable_array.append(np.mean(y[value_filter]))
            std_variable_array.append(np.std(y[value_filter]))
    
    array = np.array(mean_variable_array)
    array_error = np.array(std_variable_array)
    
    if norm:
        array_error = array_error / np.max(array)
        array = array / np.max(array)

    return values, array, array_error


def fit_hist1d(
    x, y, fit={"func": "polynomial", "trimm": 5, "print": True}, debug=False
):
    """
    Given a 1D histogram, fit a function to the histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
        func (str): function to fit to histogram (exponential, polynomial, etc.).
        trimm (int): number of bins to remove from the beginning and end of the histogram.
        debug (bool): If True, the debug mode is activated.

    Returns:
        func (function): function to fit to histogram.
    """
    # Remove x values at the beginning and end of the array
    x = x[fit["trimm"] : -fit["trimm"]]
    y = y[fit["trimm"] : -fit["trimm"]]

    if fit["func"] == "slope1":
        if fit["print"] and debug:
            rprint("Fitting line...")

        def func(x, *coefficients, debug=False):
            return slope1(x, coefficients, debug=debug)

        labels = ["Intercept"]
        initial_guess = np.random.randn(1)

    if fit["func"] == "linear":
        if fit["print"] and debug:
            rprint("Fitting line...")

        def func(x, *coefficients, debug=False):
            return linear(x, coefficients, debug=debug)

        labels = ["Slope", "Intercept"]
        initial_guess = np.random.randn(2)

    if fit["func"] == "polynomial":
        if fit["print"] and debug:
            rprint("Fitting polynomial...")

        def func(x, *coefficients, debug=False):
            return polynomial(x, coefficients, debug=debug)

        initial_guess = np.random.randn(
            3
        )  # Provide an initial guess for the polynomial coefficients
        labels = len(initial_guess) * ["coef"]

    if fit["func"] == "exponential":
        if fit["print"] and debug:
            rprint("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return exp(x, coefficients, debug=debug)

        labels = ["Amplitude", "Tau"]
        initial_guess = (1e2, 1e4)

    if fit["func"] == "exponential_offset":
        if fit["print"] and debug:
            rprint("Fitting exponential...")

        def func(x, *coefficients, debug=False):
            return exp(x, coefficients, debug=debug)

        labels = ["Amplitude", "Tau", "Offset"]
        initial_guess = (1e2, 1e4, 0)

    if fit["func"] == "gauss":
        if fit["print"] and debug:
            rprint("Fitting gaussian...")

        def func(x, *coefficients, debug=False):
            return gauss(x, coefficients, debug=debug)

        labels = ["Amplitude", "Mean", "Sigma"]
        initial_guess = (np.max(y), x[np.argmax(y)], np.std(y))
        # initial_guess = (0,0,0)

    # Fitting the polynomial line to the data
    popt, pcov = curve_fit(func, x, y, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    return func, labels, popt, perr


def get_hist2d(x, y, per:tuple=(1, 99), acc=50, norm=True, density=False, logz:bool=False, debug:bool=False):
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
    x = np.asarray(x)
    y = np.asarray(y)
    lims = np.percentile([x, y], per, axis=1, keepdims=False)

    if debug:
        rprint(
            type(x),
            x,
            "\n",
            type(y),
            y,
            "\n[cyan]INFO: Percentile limits: " + str(lims) + "[/cyan]",
        )
    reduced_x = [
        i
        for i, j in zip(x, y)
        if lims[0][0] < i < lims[1][0] and lims[0][1] < j < lims[1][1]
    ]
    reduced_y = [
        j
        for i, j in zip(x, y)
        if lims[0][0] < i < lims[1][0] and lims[0][1] < j < lims[1][1]
    ]
    # Compute the number of bins using the Freedman-Diaconis rule
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    # acc = 2 * IQR(x) / (n^(1/3))
    if type(acc) == int:
        x_array = np.linspace(np.min(reduced_x), np.max(reduced_x), acc + 1)
        y_array = np.linspace(np.min(reduced_y), np.max(reduced_y), acc + 1)
    if type(acc) == tuple:
        x_array = np.linspace(np.min(reduced_x), np.max(reduced_x), acc[0] + 1)
        y_array = np.linspace(np.min(reduced_y), np.max(reduced_y), acc[1] + 1)
    
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[x_array, y_array], density=density)
    if logz:
        h = np.log(h)
    if norm:
        h = h / (np.sum(h))
    # x, y = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
    x_centers, y_centers = (x_edges[1:] + x_edges[:-1]) / 2, (
        y_edges[1:] + y_edges[:-1]
    ) / 2
    return x_centers, y_centers, h


def get_hist2d_fit(
    x,
    y,
    fig:go.Figure,
    idx:tuple,
    per:tuple = (1, 99),
    acc:float = 50,
    fit:dict = {
        "color": "grey",
        "opacity": 1,
        "trimm": 5,
        "spec_type": "max",
        "func": "linear",
        "threshold": 0.4,
        "range":(0,10),
        "print": True,
    },
    density = None,
    logz:bool = False,
    zoom:bool = False,
    debug:bool = False,
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
    hx, hy, hz = get_hist2d(x, y, per=per, acc=acc, density=density, logz=logz, debug=debug)

    fig.add_trace(
        go.Heatmap(z=hz.T, x=hx, y=hy, coloraxis="coloraxis"), row=idx[0], col=idx[1]
    )
    if fit["spec_type"] == "intercept":
        popt, perr, labels = [], [], []
        intercepts = find_hist2d_intercept(x, y, acc, irange=fit["range"], threshold=fit["threshold"], show=False, debug=debug)
        array = np.arange(np.min(hx), np.max(hx))
        for b in intercepts:
            fig.add_trace(go.Scatter(x=array,y=array-b, mode="lines", marker=dict(color=fit["color"], opacity=fit["opacity"])), row=idx[0], col=idx[1])
            popt = np.concatenate((popt,[-b]))
            perr = np.concatenate((perr,[10/acc]))
            labels = np.concatenate((labels,["Intercept"]))

    else:
        x_spec, y_spec = spectrum_hist2d(hx, hy, hz, fit=fit, debug=debug)
        func, labels, popt, perr = fit_hist1d(x_spec, y_spec, fit=fit, debug=debug)

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
            "\nFit parameter %s: %f +/- %f" % (labels[i], popt[i], perr[i])
            for i in range(len(labels))
        ]
        rprint("[cyan]INFO: " + "".join(debug_text) + "[/cyan]")
    return fig, popt, perr


def plot_hist2d_fit(
    fig, idx, func, popt, perr, x, y, fit={"color": "grey", "opacity": 1}, debug=False
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
    if debug:
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
    per=[1, 99],
    acc=50,
    fit={"color": "grey", "trimm": 5, "func": "gauss"},
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
        func (str): function to fit to histogram (exponential, polynomial, etc.).
        trimm (int): number of bins to remove from the beginning and end of the histogram.
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly figure): plotly figure.
        popt (array): array with the fit parameters.
        perr (array): array with the fit errors.
    """
    x, h = get_hist1d(x, per=per, acc=acc, debug=debug)
    fig.add_trace(
        go.Bar(x=x, y=h, marker=dict(color="grey"), name="Spectrum"),
        row=idx[0],
        col=idx[1],
    )
    fig.update_layout(bargap=0)

    try:
        func, labels, popt, perr = fit_hist1d(x, h, fit=fit, debug=debug)
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
            "\nFit parameter %s: %f +/- %f" % (labels[i], popt[i], perr[i])
            for i in range(len(labels))
        ]
        rprint("[cyan]INFO: " + "".join(debug_text) + "[/cyan]")
    return fig, popt, perr


def find_hist2d_intercept(x, y, acc:int, irange:tuple=(0,10), threshold:float=.6, slope:float=1, show=False, debug=False)-> list:
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
    for idx,b in enumerate(np.linspace(irange[0], irange[1], acc)):
        bins, bar = np.histogram((y+b)/(slope*x), bins=acc)
        if 1-1/acc < bar[np.argmax(bins)] < 1+1/acc:
            if len(intercepts) == 0:
                intercepts.append(b)
                counts.append(bins[np.argmax(bins)])
                if show: plt.step(bar[:-1], bins, where="post", label="b = %f" % b)
            else:
                if b - intercepts[-1] < threshold:
                    if bins[np.argmax(bins)] > counts[-1]:
                        intercepts[-1] = b
                        counts[-1] = bins[np.argmax(bins)]
                        if show: plt.step(bar[:-1], bins, where="post", label="b = %f" % b)
                    else:
                        if debug: print("Skipping", b, intercepts[-1])
                else:
                    intercepts.append(b)
                    counts.append(bins[np.argmax(bins)])
                    if show: plt.step(bar[:-1], bins, where="post", label="b = %f" % b)
    
    if show:
        plt.xlabel(r"($E_{e}$+const) / $E_{\nu}$")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()
    
    # if debug:
    #     debug_text = [
    #         "\nFit parameter Intercept: %f +/- %f" % (intercepts[i],10/acc)
    #         for i in range(len(intercepts))
    #     ]
    #     rprint("[cyan]INFO: " + "".join(debug_text) + "[/cyan]")
    return intercepts