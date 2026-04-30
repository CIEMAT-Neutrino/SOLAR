import json
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.colors as colors
import matplotlib.cm as cm
import plotly.graph_objects as go

from typing import Optional
from rich import print as rprint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from matplotlib.collections import LineCollection
from plotly.validators.scatter.marker import SymbolValidator

from src.utils import get_project_root
from .lib_default import load_analysis_info

colors = px.colors.qualitative.Prism
symbols = SymbolValidator().values
root = get_project_root()

pio.templates["DUNE"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="DUNE watermark",
            text="<b>DUNE</b> Simulation",
            # textangle=-30,
            opacity=0.75,
            font=dict(color="black", size=25),
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.0,
            showarrow=False,
        )
    ]
)
pio.templates.default = "presentation"


def print_watermark(
    fig, watermark: str = "<b>DUNE</b> Work In Progress", debug: bool = False
):
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
    rows, cols = find_subplots(fig, debug=debug)

    # Add annotation inside each subplot
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            # If figure is heatmap, add annotation in white color

            fig.add_annotation(
                dict(
                    name="DUNE watermark",
                    text=watermark,
                    # textangle=-30,
                    opacity=0.75,
                    font=dict(color="black", size=25),
                    xref="x domain",
                    yref="y domain",
                    x=0.01,
                    y=1.0,
                    showarrow=False,
                ),
                row=row,
                col=col,
            )

    return fig


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
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    # Add colorbar to the plot
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Threshold")

    return ax.add_collection(lc)


def find_subplots(fig, debug: bool = False):
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

        except Exception:
            if debug:
                rprint(
                    "[red][ERROR] Method fig._get_subplot_rows_columns() not availabel![/red]"
                )
            rows, cols = 1, 1

    else:
        rows, cols = 1, 1
        rprint(f"[red][ERROR] Unknown figure type! {type(type(fig))}[/red]")

    return rows, cols


def _axis_ref_to_layout_key(axis_ref: str, axis_name: str) -> str:
    """Convert a trace axis ref (x, x2, y3) to layout axis key (xaxis, xaxis2, yaxis3)."""
    if not axis_ref:
        return f"{axis_name}axis"

    suffix = axis_ref[1:]
    return f"{axis_name}axis{suffix}" if suffix else f"{axis_name}axis"


def _to_float_1d(values):
    """Convert data to a flat float array, coercing datetime-like values when possible."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return np.array([], dtype=float)

    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    flat = arr.ravel()

    numeric = np.asarray(pd.to_numeric(flat, errors="coerce"), dtype=float)
    if np.isfinite(numeric).any():
        return numeric

    dt = pd.to_datetime(flat, errors="coerce")
    dt64 = np.asarray(dt, dtype="datetime64[ns]")
    valid_dt = ~np.isnat(dt64)
    if np.any(valid_dt):
        converted = np.full(dt64.shape, np.nan, dtype=float)
        converted[valid_dt] = dt64[valid_dt].astype("int64").astype(float)
        return converted

    return np.full(arr.size, np.nan, dtype=float)


def auto_place_legend(
    fig: go.Figure,
    box_size: tuple = (0.26, 0.18),
    pad: float = 0.01,
    watermark: bool = False,
    debug: bool = False,
):
    """
    Place legend in the least crowded corner of the figure or subplot domains.
    Avoids top-left corner when watermark is active.

    Returns:
        dict: legend layout keys (x, y, xanchor, yanchor)
    """
    axis_limits = {}
    domains = []
    paper_points = []

    # First pass: estimate axis ranges from trace data when explicit ranges are absent.
    for trace in fig.data:
        x_raw = getattr(trace, "x", None)
        y_raw = getattr(trace, "y", None)
        if x_raw is None or y_raw is None:
            continue

        x = _to_float_1d(x_raw)
        y = _to_float_1d(y_raw)
        if x.size == 0 or y.size == 0:
            continue

        x_key = _axis_ref_to_layout_key(getattr(trace, "xaxis", "x"), "x")
        y_key = _axis_ref_to_layout_key(getattr(trace, "yaxis", "y"), "y")

        x_fin = x[np.isfinite(x)]
        y_fin = y[np.isfinite(y)]
        if x_fin.size:
            if x_key not in axis_limits:
                axis_limits[x_key] = [np.min(x_fin), np.max(x_fin)]
            else:
                axis_limits[x_key][0] = min(axis_limits[x_key][0], np.min(x_fin))
                axis_limits[x_key][1] = max(axis_limits[x_key][1], np.max(x_fin))
        if y_fin.size:
            if y_key not in axis_limits:
                axis_limits[y_key] = [np.min(y_fin), np.max(y_fin)]
            else:
                axis_limits[y_key][0] = min(axis_limits[y_key][0], np.min(y_fin))
                axis_limits[y_key][1] = max(axis_limits[y_key][1], np.max(y_fin))

    # Second pass: project points into paper coordinates.
    for trace in fig.data:
        x_raw = getattr(trace, "x", None)
        y_raw = getattr(trace, "y", None)
        if x_raw is None or y_raw is None:
            continue

        x = _to_float_1d(x_raw)
        y = _to_float_1d(y_raw)
        n_points = min(x.size, y.size)
        if n_points == 0:
            continue

        x = x[:n_points]
        y = y[:n_points]

        x_key = _axis_ref_to_layout_key(getattr(trace, "xaxis", "x"), "x")
        y_key = _axis_ref_to_layout_key(getattr(trace, "yaxis", "y"), "y")

        xaxis = getattr(fig.layout, x_key, None)
        yaxis = getattr(fig.layout, y_key, None)

        x_dom = tuple(getattr(xaxis, "domain", [0, 1]) if xaxis else [0, 1])
        y_dom = tuple(getattr(yaxis, "domain", [0, 1]) if yaxis else [0, 1])
        domains.append((x_dom, y_dom))

        x_range = getattr(xaxis, "range", None) if xaxis else None
        y_range = getattr(yaxis, "range", None) if yaxis else None

        if x_range and len(x_range) == 2:
            x_range_num = _to_float_1d(x_range)
            x_min, x_max = float(np.nanmin(x_range_num)), float(np.nanmax(x_range_num))
        else:
            x_min, x_max = axis_limits.get(x_key, [np.nan, np.nan])

        if y_range and len(y_range) == 2:
            y_range_num = _to_float_1d(y_range)
            y_min, y_max = float(np.nanmin(y_range_num)), float(np.nanmax(y_range_num))
        else:
            y_min, y_max = axis_limits.get(y_key, [np.nan, np.nan])

        valid_limits = (
            np.isfinite(x_min)
            and np.isfinite(x_max)
            and np.isfinite(y_min)
            and np.isfinite(y_max)
        )
        if not valid_limits or x_max == x_min or y_max == y_min:
            continue

        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue

        nx = (x[mask] - x_min) / (x_max - x_min)
        ny = (y[mask] - y_min) / (y_max - y_min)
        keep = (nx >= 0) & (nx <= 1) & (ny >= 0) & (ny <= 1)
        nx = nx[keep]
        ny = ny[keep]

        if nx.size == 0:
            continue

        paper_x = x_dom[0] + nx * (x_dom[1] - x_dom[0])
        paper_y = y_dom[0] + ny * (y_dom[1] - y_dom[0])
        paper_points.extend(zip(paper_x, paper_y))

    if not paper_points:
        return dict(x=0.99, y=0.99, xanchor="right", yanchor="top")

    unique_domains = list(dict.fromkeys(domains)) if domains else [((0, 1), (0, 1))]
    points = np.array(paper_points)

    def score_box(x0, x1, y0, y1):
        return int(
            np.sum(
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
        )

    candidates = []
    width, height = box_size

    for x_dom, y_dom in unique_domains:
        x0, x1 = x_dom
        y0, y1 = y_dom

        # top-right
        x, y = x1 - pad, y1 - pad
        candidates.append(
            (
                score_box(max(x - width, x0), x, max(y - height, y0), y),
                0,
                dict(x=x, y=y, xanchor="right", yanchor="top"),
            )
        )
        # top-left
        x, y = x0 + pad, y1 - pad
        candidates.append(
            (
                score_box(x, min(x + width, x1), max(y - height, y0), y),
                1,
                dict(x=x, y=y, xanchor="left", yanchor="top"),
            )
        )
        # bottom-right
        x, y = x1 - pad, y0 + pad
        candidates.append(
            (
                score_box(max(x - width, x0), x, y, min(y + height, y1)),
                2,
                dict(x=x, y=y, xanchor="right", yanchor="bottom"),
            )
        )
        # bottom-left
        x, y = x0 + pad, y0 + pad
        candidates.append(
            (
                score_box(x, min(x + width, x1), y, min(y + height, y1)),
                3,
                dict(x=x, y=y, xanchor="left", yanchor="bottom"),
            )
        )

    # If watermark is active, exclude top-left corner (index 1) to avoid overlap
    if watermark:
        candidates = [c for c in candidates if c[1] != 1]

    if not candidates:
        # Fallback if all candidates filtered out (shouldn't happen)
        return dict(x=0.99, y=0.99, xanchor="right", yanchor="top")

    candidates.sort(key=lambda item: (item[0], item[1]))
    if debug:
        rprint(f"[cyan]Auto legend overlap score: {candidates[0][0]}[/cyan]")

    return candidates[0][2]


def grouped_axis_from_starts(
    energy_axis_tail: np.ndarray,
    starts: np.ndarray,
    fallback_width: float = 1.0,
):
    """Build grouped-bin centers and widths from adaptive-rebin group starts."""
    starts_array = np.asarray(starts, dtype=int)
    if starts_array.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    energy_tail = np.asarray(energy_axis_tail, dtype=float)
    ends = np.append(starts_array[1:], len(energy_tail))
    centers = []
    widths = []
    for start, end in zip(starts_array, ends):
        this_slice = np.asarray(energy_tail[start:end], dtype=float)
        centers.append(float(np.mean(this_slice)))
        if len(this_slice) > 1:
            this_diff = np.diff(this_slice)
            this_step = float(np.median(this_diff)) if len(this_diff) > 0 else float(fallback_width)
            widths.append(float(this_slice[-1] - this_slice[0] + this_step))
        else:
            widths.append(float(fallback_width))

    return np.asarray(centers, dtype=float), np.asarray(widths, dtype=float)


def add_histogram_style_legend_traces(
    fig: go.Figure,
    row: int = 1,
    col: int = 1,
    legend: Optional[str] = None,
    legendgroup: str = "linestyle",
    legendgrouptitle: str = "Histogram",
    styles: Optional[list] = None,
):
    """Add style-only legend entries (Raw/Smoothed/Rebinned) for histogram overlays."""
    if styles is None:
        styles = [
            {"name": "Raw", "color": "gray", "width": 2, "dash": "dot", "opacity": None, "showlegend": True},
            {"name": "Smoothed", "color": "gray", "width": 3, "dash": "solid", "opacity": None, "showlegend": True},
        ]

    for style in styles:
        trace = go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=str(style.get("name", "Style")),
            line=dict(
                color=style.get("color", "gray"),
                width=int(style.get("width", 2)),
                dash=style.get("dash", "solid"),
            ),
            opacity=style.get("opacity", None),
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle),
            showlegend=bool(style.get("showlegend", True)),
        )
        if legend is not None:
            trace.legend = legend
        fig.add_trace(trace, row=row, col=col)

    return fig


def add_significance_series_trace(
    fig: go.Figure,
    x,
    y,
    name_prefix: str,
    row: int = 2,
    col: int = 1,
    color: str = "black",
    width: int = 2,
    dash: str = "solid",
    legend: str = "legend2",
    legendgroup: str = "Significance",
    legendgrouptitle: str = "Significance",
    showlegend: bool = True,
    append_total: bool = True,
    total_digits: int = 1,
    line_shape: str = "hvh",
):
    """Add one significance series trace with consistent styling and optional total-σ in legend label."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total_sigma = float(np.sqrt(np.sum(np.power(y_values, 2))))

    if append_total:
        label = f"{name_prefix}: {total_sigma:.{int(total_digits)}f}"
    else:
        label = str(name_prefix)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=label,
            showlegend=showlegend,
            line=dict(color=color, width=int(width), dash=dash),
            line_shape=line_shape,
            legend=legend,
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle),
        ),
        row=row,
        col=col,
    )

    return total_sigma


def add_significance_bin_labels(
    fig: go.Figure,
    x,
    y,
    label_values=None,
    row: int = 2,
    col: int = 1,
    text_prefix: str = "",
    digits: int = 1,
    label_stride: int = 1,
    show_zero: bool = False,
    color: str = "black",
    font_size: int = 10,
    textposition: str = "top center",
    y_offset_fraction: float = 0.1,
):
    """Overlay optional bin labels at y positions, optionally formatting alternate values."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if label_values is None:
        label_display_values = y_values
    else:
        label_display_values = np.nan_to_num(
            np.asarray(label_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if len(label_display_values) != len(y_values):
            raise ValueError("label_values must have the same length as y")
    stride = max(1, int(label_stride))
    prefix = str(text_prefix)

    offset_fraction = max(0.0, float(y_offset_fraction))
    position = str(textposition).lower()
    direction = -1.0 if position.startswith("bottom") else 1.0
    scale_ref = float(np.nanmax(np.abs(y_values))) if y_values.size > 0 else 0.0
    min_offset = max(1e-6, scale_ref * 1e-3)
    delta = np.maximum(np.abs(y_values) * offset_fraction, min_offset)
    y_text_values = y_values + direction * delta

    text_values = []
    for idx, value in enumerate(label_display_values):
        if idx % stride != 0:
            text_values.append("")
            continue
        if (not show_zero) and np.isclose(value, 0.0):
            text_values.append("")
            continue
        if len(prefix) > 0:
            text_values.append(f"{prefix}{idx}:{value:.{int(digits)}f}")
        else:
            text_values.append(f"{value:.{int(digits)}f}")

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_text_values,
            mode="text",
            text=text_values,
            textposition=textposition,
            textfont=dict(size=int(font_size), color=color),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    return fig


def add_variable_width_hist_trace(
    fig: go.Figure,
    x,
    y,
    widths,
    style: str = "step",
    row: int = 1,
    col: int = 1,
    name: str = "Histogram",
    color: str = "black",
    width: int = 3,
    dash: str = "solid",
    error_y=None,
    legend: Optional[str] = None,
    legendgroup: Optional[str] = None,
    legendgrouptitle: Optional[str] = None,
    showlegend: bool = True,
    opacity: Optional[float] = None,
    bar_offsetgroup: Optional[str] = None,
    bar_alignmentgroup: Optional[str] = None,
):
    """Draw histogram-like traces for variable bin widths using line, step, or bar style."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    width_values = np.asarray(widths, dtype=float)
    if len(x_values) != len(y_values) or len(x_values) != len(width_values):
        raise ValueError("x, y, and widths must have the same length")

    error_values = None
    if error_y is not None:
        error_values = np.nan_to_num(np.asarray(error_y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    style_value = str(style).lower()
    if style_value == "bar":
        trace = go.Bar(
            x=x_values,
            y=y_values,
            width=width_values,
            offset=0,
            name=name,
            marker=dict(color=color, line=dict(color=color, width=0)),
            opacity=opacity,
            showlegend=showlegend,
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle) if legendgrouptitle is not None else None,
            offsetgroup=bar_offsetgroup,
            alignmentgroup=bar_alignmentgroup,
        )
        if error_values is not None:
            trace.error_y = dict(type="data", array=error_values)
        if legend is not None:
            trace.legend = legend
        fig.add_trace(trace, row=row, col=col)
        return fig

    if style_value == "step":
        x_left = x_values - 0.5 * width_values
        x_right = x_values + 0.5 * width_values
        x_step = np.empty(3 * len(x_values), dtype=float)
        y_step = np.empty(3 * len(y_values), dtype=float)
        x_step[0::3] = x_left
        x_step[1::3] = x_right
        x_step[2::3] = np.nan
        y_step[0::3] = y_values
        y_step[1::3] = y_values
        y_step[2::3] = np.nan

        trace = go.Scatter(
            x=x_step,
            y=y_step,
            mode="lines",
            name=name,
            line=dict(color=color, width=int(width), dash=dash),
            line_shape="linear",
            opacity=opacity,
            showlegend=showlegend,
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle) if legendgrouptitle is not None else None,
        )
        if legend is not None:
            trace.legend = legend
        fig.add_trace(trace, row=row, col=col)

        if error_values is not None:
            error_trace = go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                marker=dict(size=0, opacity=0),
                error_y=dict(type="data", array=error_values, color=color),
                showlegend=False,
                hoverinfo="skip",
            )
            if legend is not None:
                error_trace.legend = legend
            fig.add_trace(error_trace, row=row, col=col)
        return fig

    trace = go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        line=dict(color=color, width=int(width), dash=dash),
        line_shape="hvh",
        opacity=opacity,
        showlegend=showlegend,
        legendgroup=legendgroup,
        legendgrouptitle=dict(text=legendgrouptitle) if legendgrouptitle is not None else None,
    )
    if error_values is not None:
        trace.error_y = dict(type="data", array=error_values)
    if legend is not None:
        trace.legend = legend
    fig.add_trace(trace, row=row, col=col)
    return fig


def add_reference_pair_traces(
    fig: go.Figure,
    x,
    y_raw,
    y_smoothed,
    name: str,
    raw_style: dict,
    smoothed_style: dict,
    row: int = 1,
    col: int = 1,
    legend: str = "legend",
    legendgroup: str = "reference",
    legendgrouptitle: str = "Reference",
    showlegend_raw: bool = False,
    showlegend_smoothed: bool = True,
    line_shape: str = "linear",
    y_upper=None,
    y_lower=None,
    band_fillcolor: str = "rgba(68, 68, 68, 0.3)",
):
    """Add paired raw/smoothed traces for a reference curve and optional uncertainty band."""
    x_values = np.asarray(x, dtype=float)
    raw_values = np.nan_to_num(np.asarray(y_raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    smooth_values = np.nan_to_num(np.asarray(y_smoothed, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=raw_values,
            mode="lines",
            name=name,
            line=dict(
                color=raw_style.get("color", "black"),
                dash=raw_style.get("dash", "dot"),
                width=int(raw_style.get("width", 2)),
            ),
            line_shape=line_shape,
            legend=legend,
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle),
            showlegend=showlegend_raw,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=smooth_values,
            mode="lines",
            name=name,
            line=dict(
                color=smoothed_style.get("color", "black"),
                dash=smoothed_style.get("dash", "solid"),
                width=int(smoothed_style.get("width", 3)),
            ),
            line_shape=line_shape,
            legend=legend,
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legendgrouptitle),
            showlegend=showlegend_smoothed,
        ),
        row=row,
        col=col,
    )

    if y_upper is not None and y_lower is not None:
        upper_values = np.nan_to_num(np.asarray(y_upper, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        lower_values = np.nan_to_num(np.asarray(y_lower, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper_values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower_values,
                mode="lines",
                line=dict(width=0),
                fillcolor=band_fillcolor,
                fill="tonexty",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    return fig


def format_coustom_plotly(
    fig: go.Figure,
    add_units: bool = True,
    add_watermark: bool = True,
    bargap: int = 0,
    figsize: Optional[tuple] = None,
    fontsize: int = 16,
    legend: Optional[dict] = None,
    legend_title: Optional[str] = None,
    log: tuple = (False, False),
    margin: dict = {"auto": True},
    matches: tuple = ("x", "y"),
    ranges: tuple = (None, None),
    template: Optional[str] = None,
    tickformat: tuple = (".s", ".s"),
    title: Optional[str] = None,
    legend_auto: bool = True,
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
        legend_auto (bool): True to auto-place legend unless manually pinned with x and y
        debug (bool): True to print debug statements, False otherwise (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    if isinstance(title, str):
        fig.update_layout(title=title)

    default_legend = dict(
        groupclick="toggleitem",
        font=dict(size=fontsize - 3),
        bgcolor="rgba(0,0,0,0)",
        # Change the fontsize of the legendtitle
        title_font=dict(size=fontsize),
    )
    if legend == None:
        legend = default_legend
    elif isinstance(legend, dict):
        # Update legend with default values
        for key in default_legend.keys():
            if key not in legend.keys():
                legend[key] = default_legend[key]
    else:
        # Print error message
        rprint("[red][ERROR] Invalid legend type! Must be a dictionary![/red]")
        legend = default_legend

    has_manual_position = isinstance(legend, dict) and "x" in legend and "y" in legend
    if legend_auto and not has_manual_position:
        legend.update(auto_place_legend(fig, watermark=add_watermark, debug=debug))

    if figsize == None:
        rows, cols = find_subplots(fig, debug=debug)
        figsize = (800 + 400 * (cols - 1), 600 + 200 * (rows - 1))
        # if debug:
        #     rprint(f"Figure size set to {figsize[0]}x{figsize[1]}")

    default_margin = {"color": "white", "margin": (0, 0, 0, 0)}
    if margin != None:
        for key in default_margin.keys():
            if key not in margin.keys():
                margin[key] = default_margin[key]

    fig.update_layout(
        legend=legend,
        template=template,
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
    )  # tickformat=",.1s" for scientific notation

    if add_watermark:
        fig = print_watermark(fig, debug=debug)

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
    colorscale = px.colors.sequential.Turbo
    fig.update_layout(coloraxis={"colorscale": colorscale})

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


def get_units(var: str, debug: bool = False):
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
        "Time": " (us) ",
        "Energy": " (MeV) ",
        "Charge": " (ADC x tick) ",
    }
    unit = ""
    for unit_key in list(units.keys()):
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
        "mu": "\u03bc",
        "pi": "\u03c0",
        "gamma": "\u03b3",
        "Sigma": "\u03a3",
        "Lambda": "\u039b",
        "alpha": "\u03b1",
        "beta": "\u03b2",
        "gamma": "\u03b3",
        "delta": "\u03b4",
        "epsilon": "\u03b5",
        "zeta": "\u03b6",
        "eta": "\u03b7",
        "theta": "\u03b8",
        "iota": "\u03b9",
        "kappa": "\u03ba",
        "lambda": "\u03bb",
        "mu": "\u03bc",
        "nu": "\u03bd",
        "xi": "\u03be",
        "omicron": "\u03bf",
        "pi": "\u03c0",
        "rho": "\u03c1",
        "sigma": "\u03c3",
        "tau": "\u03c4",
        "upsilon": "\u03c5",
        "phi": "\u03c6",
        "chi": "\u03c7",
        "psi": "\u03c8",
        "omega": "\u03c9",
    }

    unicode_symbol = {
        "PlusMinus": "\u00b1",
        "MinusPlus": "\u2213",
        "Plus": "\u002b",
        "Minus": "\u2212",
        "Equal": "\u003d",
        "NotEqual": "\u2260",
        "LessEqual": "\u2264",
        "GreaterEqual": "\u2265",
        "Less": "\u003c",
        "Greater": "\u003e",
        "Approximately": "\u2248",
        "Proportional": "\u221d",
        "Infinity": "\u221e",
        "Degree": "\u00b0",
        "Prime": "\u2032",
        "DoublePrime": "\u2033",
        "TriplePrime": "\u2034",
        "QuadruplePrime": "\u2057",
        "Micro": "\u00b5",
        "PerMille": "\u2030",
        "Permyriad": "\u2031",
        "Minute": "\u2032",
        "Second": "\u2033",
        "Dot": "\u02d9",
        "Cross": "\u00d7",
        "Star": "\u22c6",
        "Circle": "\u25cb",
        "Square": "\u25a1",
        "Diamond": "\u25c7",
        "Triangle": "\u25b3",
        "LeftTriangle": "\u22b2",
        "RightTriangle": "\u22b3",
        "LeftTriangleEqual": "\u22b4",
        "RightTriangleEqual": "\u22b5",
        "LeftTriangleBar": "\u29cf",
        "RightTriangleBar": "\u29d0",
        "LeftTriangleEqualBar": "\u29cf",
        "RightTriangleEqualBar": "\u29d0",
        "LeftRightArrow": "\u2194",
        "UpDownArrow": "\u2195",
        "UpArrow": "\u2191",
        "DownArrow": "\u2193",
        "LeftArrow": "\u2190",
        "RightArrow": "\u2192",
        "UpArrowDownArrow": "\u21c5",
        "LeftArrowRightArrow": "\u21c4",
        "LeftArrowLeftArrow": "\u21c7",
        "UpArrowUpArrow": "\u21c8",
        "RightArrowRightArrow": "\u21c9",
        "DownArrowDownArrow": "\u21ca",
        "LeftRightVector": "\u294e",
        "RightUpDownVector": "\u294f",
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
        "1": "\u00b9",
        "2": "\u00b2",
        "3": "\u00b3",
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
        "y": "ʸ",
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
        "l": "ₗ",
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
                patches[i][j].set_facecolor(plt.cm.viridis(n[i][j] / np.max(n)))
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


# Draw the contour into a plotly figure
def draw_contour(fig, idx, label, data, color="black", dash="solid"):
    # Convert the data into a numpy array
    data = np.array(data).astype(float)

    # Create a contour plot from set of 2d points
    fig.add_trace(
        go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            name=label,
            mode="lines",
            line=dict(
                color=color,
                width=2,
                dash=dash,
            ),
            line_shape="spline",
            # marker_symbol=symbols[idx],
            # marker=dict(size=5, color=color),
            showlegend=idx == 0,
        )
    )
    # Show the figure
    return fig


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
    analysis_info = load_analysis_info(str(root))
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

            hist, edges = np.histogram(this_df[variable], bins=bins, density=density)
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
        np.sqrt((bins_error[0] / bins[0]) ** 2 + (bins_error[1] / bins[1]) ** 2)
        * residual
    )
    # Calculate the chi2 between the two histograms but only if the bin content is > 0
    chi2 = np.sum(
        (bins[0][bins[0] != 0] - bins[1][bins[0] != 0]) ** 2 / bins[0][bins[0] != 0]
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
