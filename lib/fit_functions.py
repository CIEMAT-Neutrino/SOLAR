import numpy as np
import plotly.graph_objects as go

from rich import print as rprint
from scipy.optimize import curve_fit
np.seterr(divide='ignore', invalid='ignore')

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


def get_hist1d(x, per=[1, 99], acc=50, norm=True, density=False, debug=False):
    """
    Given an x array, generate a 1D histogram.

    Args:
        x (array): x-axis array.
        acc (int): number of bins.

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
    if type(acc) == int:
        x_array = np.linspace(np.min(x), np.max(x), acc + 1)
    elif type(acc) == list or type(acc) == np.ndarray:
        x_array = acc
    else:
        rprint("[red]ERROR: acc must be an integer, list, or numpy array![/red]")
        raise ValueError

    h, x = np.histogram(x, bins=x_array, density=density)

    if norm:
        h = h / (np.sum(h))
    x = (x[1:] + x[:-1]) / 2

    return x, h


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


def get_hist2d(x, y, per=[1, 99], acc=50, norm=True, density=False, debug=False):
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
    x_array = np.linspace(np.min(reduced_x), np.max(reduced_x), acc + 1)
    y_array = np.linspace(np.min(reduced_y), np.max(reduced_y), acc + 1)
    h, x_edges, y_edges = np.histogram2d(x, y, bins=[x_array, y_array], density=density)
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
    fig,
    idx,
    per=[1, 99],
    acc=50,
    fit={
        "color": "grey",
        "opacity": 1,
        "trimm": 5,
        "spec_type": "max",
        "func": "linear",
        "threshold": 0.4,
        "print": True,
    },
    density=None,
    zoom=False,
    debug=False,
):
    """
    Given x and y arrays, generate a 2D histogram and fit a function to the histogram.

    Args:
        x (array): x-axis array.
        y (array): y-axis array.
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
    """
    x, y, h = get_hist2d(x, y, per=per, acc=acc, density=density, debug=debug)

    fig.add_trace(
        go.Heatmap(z=h.T, x=x, y=y, coloraxis="coloraxis"), row=idx[0], col=idx[1]
    )
    if fit["spec_type"] == "top+bottom":
        popt, perr = [], []
        x_spec, y_top, y_bottom = spectrum_hist2d(x, y, h, fit=fit, debug=debug)
        for spectrum, spectrum_name in zip([y_top, y_bottom], ["top", "bottom"]):
            func, labels, this_popt, this_perr = fit_hist1d(
                x_spec, spectrum, fit=fit, debug=debug
            )

            if debug:
                fig.add_trace(
                    go.Scatter(
                        x=x_spec,
                        y=spectrum,
                        mode="markers",
                        marker=dict(color=fit["color"], opacity=fit["opacity"]),
                        name=f"{spectrum_name} Spectrum",
                    ),
                    row=idx[0],
                    col=idx[1],
                )

            fig.add_trace(
                go.Scatter(
                    x=x_spec,
                    y=func(x_spec, *this_popt),
                    mode="lines",
                    marker=dict(color=fit["color"], opacity=fit["opacity"]),
                    name=f"{spectrum_name} Fit",
                    error_y=dict(type="data", array=func(x_spec, *this_perr), visible=True),
                ),
                row=idx[0],
                col=idx[1],
            )
            popt = np.concatenate((popt,this_popt))
            perr = np.concatenate((perr,this_perr))

        if zoom:
            fig.update_xaxes(range=[np.min(x), np.max(x)], row=idx[0], col=idx[1])
            fig.update_yaxes(range=[np.min(y), np.max(y)], row=idx[0], col=idx[1])

        labels = [
            f"{spectrum_name}{label}"
            for spectrum_name in ["Top", "Bottom"]
            for label in labels
        ]

    else:
        x_spec, y_spec = spectrum_hist2d(x, y, h, fit=fit, debug=debug)
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
            fig.update_xaxes(range=[np.min(x), np.max(x)], row=idx[0], col=idx[1])
            fig.update_yaxes(range=[np.min(y), np.max(y)], row=idx[0], col=idx[1])

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
            error_y=dict(type="data", array=func(x, *perr), visible=True),
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
