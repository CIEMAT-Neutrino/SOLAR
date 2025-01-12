import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import plotly.graph_objects as go

from typing import Optional
from rich import print as rprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from matplotlib.collections import LineCollection

from src.utils import get_project_root
root = get_project_root()


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack(
        (x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    # Add colorbar to the plot
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Threshold")

    return ax.add_collection(lc)


def find_subplots(fig, debug:bool = False):
    """
    Find the number of subplots in a plotly figure

    Args:
        fig (plotly.graph_objects.Figure): plotly figure
        debug (bool): True to print debug statements, False otherwise (default: False)

    Returns:
        rows (int): number of rows
        cols (int): number of columns
    """
    # Find the number of subplots
    if type(fig) == go.Figure:
        try:
            rows, cols = fig._get_subplot_rows_columns()
            rows, cols = rows[-1], cols[-1]
            if debug:
                rprint("[cyan][INFO]: Detected number of subplots: " +
                    str(rows * cols) + "[/cyan]")
        
        except Exception:
            if debug: rprint("[red][ERROR]: Method fig._get_subplot_rows_columns() not availabel![/red]")
            rows, cols = 1, 1
   
    else:
        rows, cols = 1, 1
        rprint(f"[red][ERROR]: Unknown figure type! {type(type(fig))}[/red]")
    
    return rows, cols


def format_coustom_plotly(
    fig: go.Figure,
    title: str = None,
    legend: Optional[dict] = None,
    legend_title: str = None,
    fontsize: int = 16,
    figsize: int = None,
    ranges: tuple = (None, None),
    matches: tuple = ("x", "y"),
    tickformat: tuple = (".s", ".s"),
    log: tuple = (False, False),
    margin: dict = {"auto": True},
    add_units: bool = True,
    bargap: int = 0,
    debug: bool = False,
):
    """
    Format a plotly figure

    Args:
        fig (plotly.graph_objects.Figure): plotly figure
        title (str): title of the figure (default: None)
        legend (dict): legend options (default: dict())
        fontsize (int): font size (default: 16)
        figsize (tuple): figure size (default: None)
        ranges (tuple): axis ranges (default: (None,None))
        matches (tuple): axis matches (default: ("x","y"))
        tickformat (tuple): axis tick format (default: ('.s','.s'))
        log (tuple): axis log scale (default: (False,False))
        margin (dict): figure margin (default: {"auto":True,"color":"white","margin":(0,0,0,0)})
        add_units (bool): True to add units to axis labels, False otherwise (default: False)
        debug (bool): True to print debug statements, False otherwise (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """

    if legend == None:
        legend = dict(groupclick="toggleitem", font=dict(size=fontsize-3))

    if figsize == None:
        rows, cols = find_subplots(fig, debug=debug)
        figsize = (800 + 400 * (cols - 1), 600 + 200 * (rows - 1))

    default_margin = {"color": "white", "margin": (0, 0, 0, 0)}
    if margin != None:
        for key in default_margin.keys():
            if key not in margin.keys():
                margin[key] = default_margin[key]

    fig.update_layout(
        title=title,
        legend=legend,
        template="presentation",
        font=dict(size=fontsize),
        paper_bgcolor=margin["color"],
        bargap=bargap,
        legend_title_text=legend_title,
    )  # font size and template
    fig.update_xaxes(
        matches=matches[0],
        showline=True,
        mirror="ticks",
        showgrid=True,
        minor_ticks="inside",
        tickformat=tickformat[0],
        # range=ranges[0],
    )  # tickformat=",.1s" for scientific notation

    if ranges[0] != None:
        fig.update_xaxes(range=ranges[0])
    if ranges[1] != None:
        fig.update_yaxes(range=ranges[1])

    fig.update_yaxes(
        matches=matches[1],
        showline=True,
        mirror="ticks",
        showgrid=True,
        minor_ticks="inside",
        tickformat=tickformat[1],
        # range=ranges[1],
    )  # tickformat=",.1s" for scientific notation

    if figsize != None:
        fig.update_layout(width=figsize[0], height=figsize[1])
    if log[0]:
        fig.update_xaxes(type="log", tickmode="linear")
    if log[1]:
        fig.update_yaxes(type="log", tickmode="linear")
    if margin["auto"] == False:
        fig.update_layout(
            margin=dict(
                l=margin["margin"][0],
                r=margin["margin"][1],
                t=margin["margin"][2],
                b=margin["margin"][3],
            )
        )
    # Update colorscale to viridis but with white at the bottom
    colorscale = px.colors.sequential.Turbo[::-1]
    # colorscale.append("white")
    colorscale = colorscale[::-1]
    fig.update_layout(coloraxis={'colorscale': colorscale})

    # Update axis labels to include units
    if add_units:
        try:
            fig.update_xaxes(
                title_text=fig.layout.xaxis.title.text
                + get_units(fig.layout.xaxis.title.text, debug=debug)
            )
        except AttributeError:
            pass
        try:
            fig.update_yaxes(
                title_text=fig.layout.yaxis.title.text
                + get_units(fig.layout.yaxis.title.text, debug=debug)
            )
        except AttributeError:
            pass
    
    return fig


def get_units(var:str, debug:bool=False):
    """
    Returns the units of a variable based on the variable name

    Args:
        var (str): variable name
    """
    units = {
        "R": " (cm) ",
        "X": " (cm) ",
        "Y": " (cm) ",
        "Z": " (cm) ",
        "E": " (MeV) ",
        "P": " (MeV) ",
        "K": " (MeV) ",
        "PE": " (PE) ",
        "Time": " (tick) ",
        "Energy": " (MeV) ",
        "Charge": " (ADC x tick) ",
    }
    unit = ""
    for unit_key in list(units.keys()):
        if debug:
            print("Checking for " + unit_key + " in " + var)
        if var.endswith(unit_key):
            unit = units[unit_key]
            if debug:
                print("Unit found for " + var)
    return unit


def unicode(x):
    """
    Returns the unicode character for a given string

    Args:
        x (str): string to convert to unicode
    """
    if type(x) != str:
        raise TypeError("Input must be a string")
    unicode_greek = {
        "Delta": "\u0394",
        "mu": "\u03BC",
        "pi": "\u03C0",
        "gamma": "\u03B3",
        "Sigma": "\u03A3",
        "Lambda": "\u039B",
        "alpha": "\u03B1",
        "beta": "\u03B2",
        "gamma": "\u03B3",
        "delta": "\u03B4",
        "epsilon": "\u03B5",
        "zeta": "\u03B6",
        "eta": "\u03B7",
        "theta": "\u03B8",
        "iota": "\u03B9",
        "kappa": "\u03BA",
        "lambda": "\u03BB",
        "mu": "\u03BC",
        "nu": "\u03BD",
        "xi": "\u03BE",
        "omicron": "\u03BF",
        "pi": "\u03C0",
        "rho": "\u03C1",
        "sigma": "\u03C3",
        "tau": "\u03C4",
        "upsilon": "\u03C5",
        "phi": "\u03C6",
        "chi": "\u03C7",
        "psi": "\u03C8",
        "omega": "\u03C9",
    }

    unicode_symbol = {
        "PlusMinus": "\u00B1",
        "MinusPlus": "\u2213",
        "Plus": "\u002B",
        "Minus": "\u2212",
        "Equal": "\u003D",
        "NotEqual": "\u2260",
        "LessEqual": "\u2264",
        "GreaterEqual": "\u2265",
        "Less": "\u003C",
        "Greater": "\u003E",
        "Approximately": "\u2248",
        "Proportional": "\u221D",
        "Infinity": "\u221E",
        "Degree": "\u00B0",
        "Prime": "\u2032",
        "DoublePrime": "\u2033",
        "TriplePrime": "\u2034",
        "QuadruplePrime": "\u2057",
        "Micro": "\u00B5",
        "PerMille": "\u2030",
        "Permyriad": "\u2031",
        "Minute": "\u2032",
        "Second": "\u2033",
        "Dot": "\u02D9",
        "Cross": "\u00D7",
        "Star": "\u22C6",
        "Circle": "\u25CB",
        "Square": "\u25A1",
        "Diamond": "\u25C7",
        "Triangle": "\u25B3",
        "LeftTriangle": "\u22B2",
        "RightTriangle": "\u22B3",
        "LeftTriangleEqual": "\u22B4",
        "RightTriangleEqual": "\u22B5",
        "LeftTriangleBar": "\u29CF",
        "RightTriangleBar": "\u29D0",
        "LeftTriangleEqualBar": "\u29CF",
        "RightTriangleEqualBar": "\u29D0",
        "LeftRightArrow": "\u2194",
        "UpDownArrow": "\u2195",
        "UpArrow": "\u2191",
        "DownArrow": "\u2193",
        "LeftArrow": "\u2190",
        "RightArrow": "\u2192",
        "UpArrowDownArrow": "\u21C5",
        "LeftArrowRightArrow": "\u21C4",
        "LeftArrowLeftArrow": "\u21C7",
        "UpArrowUpArrow": "\u21C8",
        "RightArrowRightArrow": "\u21C9",
        "DownArrowDownArrow": "\u21CA",
        "LeftRightVector": "\u294E",
        "RightUpDownVector": "\u294F",
        "DownLeftRightVector": "\u2950",
        "LeftUpDownVector": "\u2951",
        "LeftVectorBar": "\u2952",
        "RightVectorBar": "\u2953",
        "RightUpVectorBar": "\u2954",
        "RightDownVectorBar": "\u2955",
    }

    unicode_dict = {**unicode_greek, **unicode_symbol}
    return unicode_dict[x]


def superscript(x):
    """
    Returns the suffix for a given string

    Args:
        x (str): string to convert to suffix
    """
    if type(x) != str:
        x = str(x)

    suffix_dict = {
        "1": "\u00B9",
        "2": "\u00B2",
        "3": "\u00B3",
        "4": "\u2074",
        "5": "\u2075",
        "6": "\u2076",
        "7": "\u2077",
        "8": "\u2078",
        "9": "\u2079",
        "0": "\u2070",
        "n": "ⁿ",
        "i": "ⁱ",
        "g": "ᵍ",
        "h": "ʰ",
        "t": "ᵗ",
        "d": "ᵈ",
        "a": "ᵃ",
        "y": "ʸ"
    }
    suffix_string = ""
    for i in range(len(x)):
        suffix_string += suffix_dict[x[i]]
    return suffix_string


def subscript(x):
    """
    Returns the suffix for a given string

    Args:
        x (str): string to convert to suffix
    """
    if type(x) != str:
        x = str(x)

    suffix_dict = {
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "0": "₀",
        "a": "ₐ",
        "e": "ₑ",
        "h": "ₕ",
        "i": "ᵢ",
        "j": "ⱼ",
        "k": "ₖ",
        "l": "ₗ"
    }
    suffix_string = ""
    for i in range(len(x)):
        suffix_string += suffix_dict[x[i]]
    return suffix_string


def update_legend(fig, dict, debug=False):
    """
    Update the legend of a plotly figure.
    """
    fig.for_each_trace(
        lambda t: t.update(
            name=dict[t.name],
            legendgroup=dict[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, dict[t.name]),
        )
    )
    return fig


def change_hist_color(n, patches, logy=False):
    """
    Change the color of a plt.hist based on the bin content

    Args:
        n (array): bin content
        patches (array): histogram patches
        logy (bool): True if log scale, False otherwise (default: False)

    Returns:
        patches (array): histogram patches with updated color
    """
    try:
        for i in range(len(n)):
            n[i] = n[i].astype(
                "int"
            )  # it MUST be integer# Good old loop. Choose colormap of your taste
            for j in range(len(patches[i])):
                patches[i][j].set_facecolor(
                    plt.cm.viridis(n[i][j] / np.max(n)))
                patches[i][j].set_edgecolor("k")
        return patches

    except:
        n = np.array(n).astype(
            "int"
        )  # it MUST be integer# Good old loop. Choose colormap of your taste
        for j in range(len(patches)):
            if logy == True:
                patches[j].set_facecolor(
                    plt.cm.viridis(np.log10(n[j]) / np.log10(np.max(n)))
                )
            else:
                patches[j].set_facecolor(plt.cm.viridis(n[j] / np.max(n)))
            patches[j].set_edgecolor("k")
        return patches


def draw_hist_colorbar(fig, n, ax, logy=False, pos="right", size="5%", pad=0.05):
    """
    Draw a colorbar for a histogram or a set of histograms

    Args:
        fig (matplotlib.figure.Figure): matplotlib figure
        n (array): bin content
        ax (matplotlib.axes.Axes): matplotlib axes
        logy (bool): True if log scale, False otherwise (default: False)
        pos (str): position of the colorbar (default: "right")
        size (str): size of the colorbar (default: "5%")
        pad (float): padding of the colorbar (default: 0.05)
    """
    cNorm = colors.Normalize(vmin=0, vmax=np.max(n))
    if logy:
        cNorm = colors.LogNorm(vmin=1, vmax=np.max(n))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if pos == "left":
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position(pos)
    fig.colorbar(cm.ScalarMappable(norm=cNorm, cmap=cm.viridis), cax=cax)


def draw_hist2d_colorbar(fig, h, ax, pos="right", size="5%", pad=0.05):
    """
    Draw a colorbar for a 2D histogram or a set of 2D histograms
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if pos == "left":
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position(pos)
    fig.colorbar(h[3], ax=ax, cax=cax)


def get_common_colorbar(data_list, bins):
    """
    Get the common colorbar for a set of histograms
    """
    for idx, data in enumerate(data_list):
        # Calculate histogram values
        hist, bins = np.histogram(data[data != 0], bins=bins)
        if idx == 0:
            max_hist = np.max(hist)
            min_hist = np.min(hist)
        else:
            if np.max(hist) > max_hist:
                max_hist = np.max(hist)
            if np.min(hist) < min_hist:
                min_hist = np.min(hist)

    return max_hist, min_hist


def plot_nhit_energy_scan(df, variable, bins=100, density=False):
    plot_list = []
    # Get energy bins from config file
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))
    energy_edges = np.linspace(
        analysis_info["REDUCED_RECO_ENERGY_RANGE"][0],
        analysis_info["REDUCED_RECO_ENERGY_RANGE"][1],
        analysis_info["REDUCED_RECO_ENERGY_BINS"] + 1,
    )
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2
    ebin = energy_edges[1] - energy_edges[0]
    for nhits in analysis_info["NHITS"]:
        for energy in energy_centers:
            this_df = df[
                (df["SignalParticleE"] > energy - ebin / 2)
                & (df["SignalParticleE"] < energy + ebin / 2)
                & (df["NHits"] >= nhits)
            ]

            hist, edges = np.histogram(
                this_df[variable], bins=bins, density=density)
            edge_centers = (edges[1:] + edges[:-1]) / 2
            plot_list.append(
                {
                    "Energy": energy,
                    "NHits": nhits,
                    variable + "Count": hist,
                    variable: edge_centers,
                }
            )
    plot_df = pd.DataFrame(plot_list)
    plot_df = plot_df.explode([variable + "Count", variable])
    fig = px.line(
        plot_df,
        x=variable,
        y=variable + "Count",
        facet_col="Energy",
        facet_col_wrap=4,
        line_shape="hvh",
        color="NHits",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )

    return fig


def histogram_comparison(
    df,
    variable,
    discriminator,
    show_residual=False,
    binning="auto",
    hist_error="binomial",
    norm="none",
    coustom_norm={},
    debug=False,
):
    """
    Compare two histograms of the same variable with different discriminator & plot the residual

    Args:
        df (pandas.DataFrame): pandas DataFrame containing the data
        variable (str): variable to plot
        discriminator (str): discriminator to compare
        show_residual (bool): True to show the residual, False otherwise (default: False)
        binning (str): binning method (default: auto)
        hist_error (str): histogram error calculation method (default: binomial)
        norm (str): histogram normalisation method (default: none)
        coustom_norm (dict): coustom normalisation factor for each discriminator (default: {})
        debug (bool): True to print debug statements, False otherwise (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    # Generate a residual plot from the histograms defined above
    discriminator_list = df[discriminator].unique()
    if len(discriminator_list) != 2:
        print("Error: discriminator must have 2 values")
        return

    # Initialize lists of size 2 for the histograms
    bins = np.empty(2, dtype=object)
    bins_error = np.empty(2, dtype=object)

    # Compute optimum number of bins for the histogram based on the number of entries
    if binning == "sturges":
        if debug:
            print("Using Sturges' formula for binning")
        # Sturges' formula
        nbins = int(np.ceil(np.log2(len(df[variable])) + 1))
    if binning == "sqrt":
        if debug:
            print("Using square root rule for binning")
        nbins = int(np.ceil(np.sqrt(len(df[variable]))))
    if binning == "fd":
        if debug:
            print("Using Freedman-Diaconis' rule for binning")
        nbins = int(
            np.ceil(
                (np.max(df[variable]) - np.min(df[variable]))
                / (
                    2
                    * (
                        np.percentile(df[variable], 75)
                        - np.percentile(df[variable], 25)
                    )
                    * np.cbrt(len(df[variable]))
                )
            )
        )
    if binning == "scott":
        if debug:
            print("Using Scott's rule for binning")
        nbins = int(
            np.ceil(
                (np.max(df[variable]) - np.min(df[variable]))
                / (3.5 * np.std(df[variable]) / np.cbrt(len(df[variable])))
            )
        )
    if binning == "doane":
        if debug:
            print("Using Doane's formula for binning")
        nbins = int(
            np.ceil(
                1
                + np.log2(len(df[variable]))
                + np.log2(
                    1
                    + np.abs((np.mean(df[variable]) - np.median(df[variable])))
                    / np.std(df[variable])
                    / np.sqrt(6)
                )
            )
        )
    else:
        if debug:
            print("Defaulting to binning with Rice rule")
        nbins = int(np.ceil(np.cbrt(len(df[variable]))))  # Rice rule

    # Generate the histograms
    nbins_min = np.min(df[variable])
    nbins_max = np.max(df[variable])
    bin_array = np.linspace(nbins_min, nbins_max, nbins)

    for i, this_discriminator in enumerate(discriminator_list):
        bins[i], edges = np.histogram(
            df[variable][df[discriminator] == this_discriminator],
            bins=bin_array,
            density=False,
        )

        # Compute normalisation factor
        if norm == "integral":
            norm_factor = np.sum(bins[i])
        if norm == "max":
            norm_factor = np.max(bins[i])
        if norm == "none":
            norm_factor = 1
        if norm == "coustom":
            norm_factor = coustom_norm[this_discriminator]

        bins[i] = bins[i] / norm_factor

        # Calculate the error on the histogram
        if hist_error == "binomial":
            bins_error[i] = bins[i] / np.sqrt(
                len(df[variable][df[discriminator] == this_discriminator]) / nbins
            )
        if hist_error == "poisson":
            bins_error[i] = np.sqrt(bins[i]) / len(
                df[variable][df[discriminator] == this_discriminator]
            )

    # Calculate the residual between the two histograms & the error
    residual = (bins[0] - bins[1]) / bins[0]
    residual_error = (
        np.sqrt((bins_error[0] / bins[0]) ** 2 +
                (bins_error[1] / bins[1]) ** 2)
        * residual
    )
    # Calculate the chi2 between the two histograms but only if the bin content is > 0
    chi2 = np.sum(
        (bins[0][bins[0] != 0] - bins[1]
         [bins[0] != 0]) ** 2 / bins[0][bins[0] != 0]
    )

    # Plot the histograms & the residual
    if show_residual:
        fig = make_subplots(
            rows=2,
            cols=1,
            print_grid=True,
            vertical_spacing=0.1,
            shared_xaxes=True,
            subplot_titles=("Histogram", ""),
            x_title=variable,
            row_heights=[0.8, 0.2],
        )
        fig.add_trace(
            go.Scatter(
                x=bin_array,
                y=residual,
                mode="markers",
                name="Residual",
                error_y=dict(array=residual_error),
                marker=dict(color="gray"),
            ),
            row=2,
            col=1,
        )
        for i in range(len(discriminator_list)):
            fig.add_trace(
                go.Bar(
                    x=bin_array,
                    y=bins[i],
                    name=discriminator_list[i],
                    opacity=0.5,
                    error_y=dict(array=bins_error[i]),
                ),
                row=1,
                col=1,
            )
        fig.add_hline(
            y=0, line_width=1, line_dash="dash", line_color="black", row=2, col=1
        )

    else:
        fig = go.Figure()
        for i in range(len(discriminator_list)):
            fig.add_trace(
                go.Bar(
                    x=bin_array,
                    y=bins[i],
                    name=discriminator_list[i],
                    opacity=0.5,
                    error_y=dict(array=bins_error[i]),
                )
            )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text="Chi2 = %.2E" % (chi2),
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_layout(showlegend=True)
    fig.update_layout(bargap=0, barmode="overlay")

    # fig.update_xaxes(title_text=fig.layout.xaxis.title.text+get_units(fig.layout.xaxis.title.text,debug=debug))
    # fig.update_yaxes(title_text=fig.layout.yaxis.title.text+get_units(fig.layout.yaxis.title.text,debug=debug))
    if debug:
        print("Histogram comparison done")
    return fig
