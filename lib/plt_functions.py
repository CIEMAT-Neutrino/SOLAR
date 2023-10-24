import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import plotly.graph_objects as go

from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots

def change_hist_color(n,patches,logy=False):
    try:
        for i in range(len(n)):
            n[i] = n[i].astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
            for j in range(len(patches[i])):
                patches[i][j].set_facecolor(plt.cm.viridis(n[i][j]/np.max(n)))
                patches[i][j].set_edgecolor("k")
        return patches
    
    except:
        n = np.array(n).astype('int') # it MUST be integer# Good old loop. Choose colormap of your taste
        for j in range(len(patches)):
            if logy == True:
                patches[j].set_facecolor(plt.cm.viridis(np.log10(n[j])/np.log10(np.max(n))))
            else:
                patches[j].set_facecolor(plt.cm.viridis(n[j]/np.max(n)))
            patches[j].set_edgecolor("k")
        return patches

def draw_hist_colorbar(fig,n,ax,logy=False,pos="right",size="5%",pad=0.05):
    cNorm = colors.Normalize(vmin=0, vmax=np.max(n))
    if logy: cNorm = colors.LogNorm(vmin=1, vmax=np.max(n))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if pos == "left":
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position(pos)
    fig.colorbar(cm.ScalarMappable(norm=cNorm,cmap=cm.viridis),cax=cax)

def draw_hist2d_colorbar(fig,h,ax,pos="right",size="5%",pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if pos == "left":
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position(pos)
    fig.colorbar(h[3],ax=ax,cax=cax)

def get_common_colorbar(data_list,bins):
    for idx,data in enumerate(data_list):
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

def format_coustom_plotly(fig,
    title="",
    fontsize=16,
    figsize=None,
    ranges=(None,None),
    matches=(None,None),
    tickformat=('.s','.s'),
    log=(False,False),
    margin={"auto":True,"color":"white","margin":(0,0,0,0)},
    add_units=False,
    debug=False):
    '''
    Format a plotly figure
    VARIABLES:
        \n fig: plotly figure
        \n fontsize: (int) font size
        \n figsize: (tuple) figure size
        \n ranges: (tuple of 2 lists) axis ranges
        \n tickformat: (tuple of 2 strings) axis tick format
        \n log: (tuple of 2 bool) axis log scale
        \n margin: (dict) margin settings
        \n add_units: (bool) add units to axis labels
        \n debug: (bool) debug mode
    '''
    fig.update_layout(title=title,template="presentation",font=dict(size=fontsize),paper_bgcolor=margin["color"]) # font size and template
    
    fig.update_xaxes(matches=matches[0],showline=True,
        mirror="ticks",showgrid=True,minor_ticks="inside",tickformat=tickformat[0],range=ranges[0]) # tickformat=",.1s" for scientific notation
    
    fig.update_yaxes(matches=matches[1],showline=True,
        mirror="ticks",showgrid=True,minor_ticks="inside",tickformat=tickformat[1],range=ranges[1]) # tickformat=",.1s" for scientific notation
    
    if figsize != None:
        fig.update_layout(width=figsize[0],height=figsize[1])
    if log[0]:
        fig.update_xaxes(type="log",tickmode="linear")
    if log[1]:
        fig.update_yaxes(type="log",tickmode="linear")
    if margin["auto"] == False:
        fig.update_layout(margin=dict(l=margin["margin"][0], r=margin["margin"][1], t=margin["margin"][2], b=margin["margin"][3]))
    # Update axis labels to include units
    if add_units:
        fig.update_xaxes(title_text=fig.layout.xaxis.title.text+get_units(fig.layout.xaxis.title.text,debug=debug))
        fig.update_yaxes(title_text=fig.layout.yaxis.title.text+get_units(fig.layout.yaxis.title.text,debug=debug))
    return fig

def histogram_comparison(df,variable,discriminator,show_residual=False,binning="auto",
    hist_error="binomial",norm="none",coustom_norm={},debug=False):
    '''
    Compare two histograms of the same variable with different discriminator & plot the residual
    VARIABLES:
        \n df: (pandas dataframe) dataframe containing the data
        \n variable: (string) variable to plot
        \n discriminator: (string) discriminator to plot
        \n binning_mode: (string) binning mode
        \n debug: (bool) debug mode
    '''
    # Generate a residual plot from the histograms defined above
    discriminator_list = df[discriminator].unique()
    if len(discriminator_list) != 2:
        print("Error: discriminator must have 2 values")
        return
    
    # Initialize lists of size 2 for the histograms
    bins = np.empty(2,dtype=object)
    bins_error = np.empty(2,dtype=object)

    # Compute optimum number of bins for the histogram based on the number of entries
    if binning == "sturges":
        if debug: print("Using Sturges' formula for binning")
        nbins = int(np.ceil(np.log2(len(df[variable]))+1)) # Sturges' formula
    if binning == "sqrt":
        if debug: print("Using square root rule for binning")
        nbins = int(np.ceil(np.sqrt(len(df[variable]))))
    if binning == "fd":
        if debug: print("Using Freedman-Diaconis' rule for binning")
        nbins = int(np.ceil((np.max(df[variable])-np.min(df[variable]))/(2*(np.percentile(df[variable],75)-np.percentile(df[variable],25))*np.cbrt(len(df[variable])))))
    if binning == "scott":
        if debug: print("Using Scott's rule for binning")
        nbins = int(np.ceil((np.max(df[variable])-np.min(df[variable]))/(3.5*np.std(df[variable])/np.cbrt(len(df[variable])))))
    if binning == "doane":
        if debug: print("Using Doane's formula for binning")
        nbins = int(np.ceil(1+np.log2(len(df[variable]))+np.log2(1+np.abs((np.mean(df[variable])-np.median(df[variable])))/np.std(df[variable])/np.sqrt(6))))
    else:
        if debug: print("Defaulting to binning with Rice rule")
        nbins = int(np.ceil(np.cbrt(len(df[variable])))) # Rice rule
    
    # Generate the histograms
    nbins_min = np.min(df[variable])
    nbins_max = np.max(df[variable])
    bin_array = np.linspace(nbins_min,nbins_max,nbins)
    
    for i,this_discriminator in enumerate(discriminator_list):
        bins[i],edges = np.histogram(df[variable][df[discriminator]==this_discriminator],bins=bin_array,density=False)
    
        # Compute normalisation factor
        if norm == "integral":
            norm_factor = np.sum(bins[i])
        if norm == "max":
            norm_factor = np.max(bins[i])
        if norm == "none":
            norm_factor = 1
        if norm == "coustom":
            norm_factor = coustom_norm[this_discriminator]
        
        bins[i] = bins[i]/norm_factor
        # print(norm_factor)
        # Calculate the error on the histogram
        if hist_error == "binomial":
            bins_error[i] = bins[i]/np.sqrt(len(df[variable][df[discriminator]==this_discriminator])/nbins)
        if hist_error == "poisson":
            bins_error[i] = np.sqrt(bins[i])/len(df[variable][df[discriminator]==this_discriminator])
        
    # Calculate the residual between the two histograms & the error
    residual = (bins[0] - bins[1])/bins[0]
    residual_error = np.sqrt((bins_error[0]/bins[0])**2+(bins_error[1]/bins[1])**2)*residual
    # Calculate the chi2 between the two histograms but only if the bin content is > 0
    chi2 = np.sum((bins[0][bins[0] != 0] - bins[1][bins[0] != 0])**2/bins[0][bins[0] != 0])     

    # Plot the histograms & the residual
    if show_residual:
        fig = make_subplots(rows=2, cols=1, print_grid=True,vertical_spacing=0.1,shared_xaxes=True,subplot_titles=("Histogram",""),x_title=variable,row_heights=[0.8,0.2])
        fig.add_trace(go.Scatter(x=bin_array,y=residual,mode="markers",name="Residual",error_y=dict(array=residual_error),marker=dict(color="gray")),row=2,col=1)
        for i in range(len(discriminator_list)):
            fig.add_trace(go.Bar(x=bin_array,y=bins[i],name=discriminator_list[i],opacity=0.5,error_y=dict(array=bins_error[i])),row=1,col=1)
        fig.add_hline(y=0, line_width=1, line_dash="dash",line_color="black",row=2,col=1)
    
    else:
        fig = go.Figure()
        for i in range(len(discriminator_list)):
            fig.add_trace(go.Bar(x=bin_array,y=bins[i],name=discriminator_list[i],opacity=0.5,error_y=dict(array=bins_error[i])))
        fig.add_hline(y=0, line_width=1, line_dash="dash",line_color="black")
        
    fig.add_annotation(x=0.01,y=0.99,xref="paper",yref="paper",text="Chi2 = %.2E"%(chi2),showarrow=False,font=dict(size=16))

    fig.update_layout(showlegend=True)
    fig.update_layout(bargap=0,barmode="overlay")
    
    # fig.update_xaxes(title_text=fig.layout.xaxis.title.text+get_units(fig.layout.xaxis.title.text,debug=debug))
    # fig.update_yaxes(title_text=fig.layout.yaxis.title.text+get_units(fig.layout.yaxis.title.text,debug=debug))
    if debug: print("Histogram comparison done")
    return fig

def get_units(var,debug=True):
    '''
    Returns the units of a variable based on the variable name
    VARIABLES:
        \n var: (string) variable name
        \n debug: (bool) debug mode
    '''
    units = {"R":" [cm] ","X":" [cm] ","Y":" [cm] ","Z":" [cm] ","Time":" [tick] ","Energy":" [MeV] ", "Charge": " [ADC x tick] "}
    for unit_key in list(units.keys()):
        if debug: print("Checking for "+unit_key +" in "+var)
        if var.endswith(unit_key):
            unit = units[unit_key]
            if debug: print("Unit found for "+var)
            break
        else:
            if debug: print("No unit found for "+var)
            unit = ""
    return unit

def unicode(x):
    unicode_greek  = {"Delta":"\u0394","mu":"\u03BC","pi":"\u03C0","gamma":"\u03B3","Sigma":"\u03A3","Lambda":"\u039B",
        "alpha":"\u03B1","beta":"\u03B2","gamma":"\u03B3","delta":"\u03B4","epsilon":"\u03B5","zeta":"\u03B6","eta":"\u03B7",
        "theta":"\u03B8","iota":"\u03B9","kappa":"\u03BA","lambda":"\u03BB","mu":"\u03BC","nu":"\u03BD","xi":"\u03BE",
        "omicron":"\u03BF","pi":"\u03C0","rho":"\u03C1","sigma":"\u03C3","tau":"\u03C4","upsilon":"\u03C5","phi":"\u03C6",
        "chi":"\u03C7","psi":"\u03C8","omega":"\u03C9"}
    
    unicode_symbol = {"PlusMinus":"\u00B1","MinusPlus":"\u2213","Plus":"\u002B","Minus":"\u2212","Equal":"\u003D","NotEqual":"\u2260",
        "LessEqual":"\u2264","GreaterEqual":"\u2265","Less":"\u003C","Greater":"\u003E","Approximately":"\u2248","Proportional":"\u221D",
        "Infinity":"\u221E","Degree":"\u00B0","Prime":"\u2032","DoublePrime":"\u2033","TriplePrime":"\u2034","QuadruplePrime":"\u2057",
        "Micro":"\u00B5","PerMille":"\u2030","Permyriad":"\u2031","Minute":"\u2032","Second":"\u2033","Dot":"\u02D9","Cross":"\u00D7",
        "Star":"\u22C6","Circle":"\u25CB","Square":"\u25A1","Diamond":"\u25C7","Triangle":"\u25B3","LeftTriangle":"\u22B2",
        "RightTriangle":"\u22B3","LeftTriangleEqual":"\u22B4","RightTriangleEqual":"\u22B5","LeftTriangleBar":"\u29CF",
        "RightTriangleBar":"\u29D0","LeftTriangleEqualBar":"\u29CF","RightTriangleEqualBar":"\u29D0","LeftRightArrow":"\u2194",
        "UpDownArrow":"\u2195","UpArrow":"\u2191","DownArrow":"\u2193","LeftArrow":"\u2190","RightArrow":"\u2192","UpArrowDownArrow":"\u21C5",
        "LeftArrowRightArrow":"\u21C4","LeftArrowLeftArrow":"\u21C7","UpArrowUpArrow":"\u21C8","RightArrowRightArrow":"\u21C9",
        "DownArrowDownArrow":"\u21CA","LeftRightVector":"\u294E","RightUpDownVector":"\u294F","DownLeftRightVector":"\u2950",
        "LeftUpDownVector":"\u2951","LeftVectorBar":"\u2952","RightVectorBar":"\u2953","RightUpVectorBar":"\u2954","RightDownVectorBar":"\u2955"}
    
    unicode_dict = {**unicode_greek,**unicode_symbol}
    return unicode_dict[x]

def update_legend(fig,dict):
    fig.for_each_trace(lambda t: t.update(name = dict[t.name],legendgroup = dict[t.name],hovertemplate = t.hovertemplate.replace(t.name, dict[t.name])))
    return fig