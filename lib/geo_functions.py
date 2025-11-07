import plotly.graph_objects as go


def add_geometry_planes(fig, geo, unzoom: float = 1, row=None, col=None, debug=False):
    """
    Function to add the TPC planes to the 3D plot of a given geometry.

    Args:
        fig (plotly.graph_objects.Figure): Figure object to add the planes to.
        geo (str): Geometry to add the planes for. Either "hd" or "vd".
        row (int): Row of the subplot to add the planes to.
        col (int): Column of the subplot to add the planes to.
        debug (bool): Print debug information.

    Returns:
        plotly.graph_objects.Figure: Figure object with the planes added.
    """
    if geo == "hd":
        fig.add_trace(
            go.Surface(
                x=[[-350, -350], [-350, -350]],
                y=[[-600, 600], [-600, 600]],
                z=[[0, 0], [1400, 1400]],
                opacity=0.25,
                name="CPA",
                text="CPA",
                colorscale=[[0, "orange"], [1, "orange"]],
                showscale=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Surface(
                x=[[350, 350], [350, 350]],
                y=[[-600, 600], [-600, 600]],
                z=[[0, 0], [1400, 1400]],
                opacity=0.25,
                name="CPA",
                text="CPA",
                colorscale=[[0, "orange"], [1, "orange"]],
                showscale=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Surface(
                x=[[0, 0], [0, 0]],
                y=[[-600, 600], [-600, 600]],
                z=[[0, 0], [1400, 1400]],
                opacity=0.25,
                name="APA",
                text="APA",
                colorscale=[[0, "green"], [1, "green"]],
                showscale=False,
            ),
            row=row,
            col=col,
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            # center=dict(x=-0.25, y=0, z=0),
            # eye=dict(x=0.25, y=1, z=1)
        )
        fig.update_layout(scene_aspectmode="auto", scene_camera=camera)

    if geo == "vd":
        fig.add_trace(
            go.Surface(
                x=[[-315, -315], [-315, -315]],
                y=[[-675, 675], [-675, 675]],
                z=[[0, 0], [2100, 2100]],
                opacity=0.25,
                name="CPA",
                text="CPA",
                colorscale=[[0, "orange"], [1, "orange"]],
                showscale=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Surface(
                x=[[315, 315], [315, 315]],
                y=[[-675, 675], [-675, 675]],
                z=[[0, 0], [2100, 2100]],
                opacity=0.25,
                name="APA",
                text="APA",
                colorscale=[[0, "green"], [1, "green"]],
                showscale=False,
            ),
            row=row,
            col=col,
        )

        camera = dict(
            up=dict(x=1, y=0.5, z=0.5),
            center=dict(x=-0.35, y=0, z=0),
            eye=dict(x=0.25 * unzoom, y=1 * unzoom, z=1 * unzoom),
        )  # Unzoom 20%

        fig.update_layout(scene_aspectmode="auto", scene_camera=camera)

    if debug:
        print("Geometry planes added")

    return fig
