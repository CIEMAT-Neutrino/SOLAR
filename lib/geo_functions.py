import plotly.graph_objects as go

def add_geometry_planes(fig, geo, debug=False):
    if geo == "hd":
        fig.add_trace(go.Surface(x=[[-350,-350],[-350,-350]],y=[[-600, 600], [-600, 600]],z=[[0, 0], [1400, 1400]],opacity=0.25,name="CPA",text="CPA",colorscale=[[0,'orange'], [1,'orange']],showscale=False))
        fig.add_trace(go.Surface(x=[[350,350],[350,350]],    y=[[-600, 600], [-600, 600]],z=[[0, 0], [1400, 1400]],opacity=0.25,name="CPA",text="CPA",colorscale=[[0,'orange'], [1,'orange']],showscale=False))
        fig.add_trace(go.Surface(x=[[0, 0], [0, 0]],         y=[[-600, 600], [-600, 600]],z=[[0, 0], [1400, 1400]],opacity=0.25,name="APA",text="APA",colorscale=[[0, 'green'], [1, 'green']],showscale=False))
    if debug: print("Geometry planes added")
    return fig