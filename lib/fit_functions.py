import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from .io_functions import print_colored

def exp(x, coefficients, debug=False):
    a = coefficients[0]
    tau = coefficients[1]
    return a*np.exp(-x/tau)

def gauss(x, coefficients, debug=False):
    a = coefficients[0]
    x0 = coefficients[1]
    sigma = coefficients[2]
    # return a/(sigma*math.sqrt(2*math.pi))*np.exp(-0.5*np.power((x-x0)/sigma,2))
    return a*np.exp(-0.5*np.power((x-x0)/sigma,2))

def quadratic(x, coefficients, debug=False):
    a = coefficients[0]
    n = coefficients[1]
    return a*np.power(x,2)+n

def linear(x, coefficients, debug=False):
    m = coefficients[0]
    n = coefficients[1]
    return m*np.asarray(x)+n

def polynomial_line(x, coefficients, debug=False):
    if debug: print("Polynomial coefficients: ",coefficients)
    return np.polyval(coefficients, x)

def generate_data(x_range, num_points, amplitude, noise_level, debug=False):
    x = np.linspace(x_range[0], x_range[1], num_points)
    polynomial_coefficients = np.random.randn(3)  # Generating random coefficients for the polynomial line (change the number for higher degree polynomials)
    y_polynomial = polynomial_line(x, polynomial_coefficients)
    noise = noise_level * np.random.randn(num_points)  # Adding random noise to the polynomial line
    z = amplitude * np.ones(num_points)  # Constant amplitude
    return x, y_polynomial + noise, z

def fit_hist2d(x, y, z, func="polynomial",debug=False):
    if x.shape != z.shape:
        print("\nFlattening 2D histogram...")
        x,y,z = flatten_hist2d(x, y, z, debug=debug)

    if func == "polynomial":
        print("Fitting polynomial...")
        def func(x, *coefficients, debug=False):
            return z + polynomial_line(x, coefficients, debug=debug)
        initial_guess = np.random.randn(3)  # Provide an initial guess for the polynomial coefficients

    if func == "exponential":
        print("Fitting exponential...")
        def func(x, *coefficients,debug=False):
            return z + exp(x, coefficients, debug=debug)
        initial_guess = (1e2,1e4)  # Provide an initial guess for the exponential coefficients
    
    # Fitting the polynomial line to the data
    popt, _ = curve_fit(func, x, y, p0=initial_guess)

    return popt

def flatten_hist2d(x, y, z, debug=False):
    '''
    Flatten a 2D histogram into a 1D array and extend the x and y arrays to match the flattened array.
    **VARIABLES:**
    \n ** - x:** x-axis array.
    \n ** - y:** y-axis array.
    \n ** - z:** 2D histogram array.
    '''
    # Print initial shapes of arrays
    if debug: print("Initial arrays (x,y,z):",x.shape,y.shape,z.shape, sep=" ")
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    x = np.repeat(x,z.shape[0])
    y = np.tile(y,z.shape[1])
    # Check if the arrays are the same length
    if len(x) != len(y):
        print("x and y arrays are not the same length!")
        print("x: ",len(x),"\ny: ",len(y))
        raise ValueError
    
    z = z.flatten()
    # Check if the arrays are the same length
    if len(x) != len(z):
        print("x and z arrays are not the same length!")
        print("x: ",len(x),"\nz: ",len(z))
        raise ValueError
    
    if debug: print("Flattened arrays (x,y,z):",len(x), len(y), len(z), sep=" ")
    return x,y,z

def spectrum_hist2d(x, y, z, spec_type="max", debug=False):
    '''
    Given a 2D histogram, return the spectrum of the histogram.
    **VARIABLES:**
    \n ** - x:** x-axis array.
    \n ** - y:** y-axis array.
    \n ** - z:** 2D histogram array.
    '''
    # Check shape of z array
    if len(z.shape) != 2:
        print("z array is not 2D!")
        print("z.shape: ",z.shape)
        raise ValueError

    # Print initial shapes of arrays
    # if debug: print("Initial arrays (x,y,z):",x.shape,y.shape,z.shape, sep=" ")
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    # Find the argmax z value along the y-axis
    z_max = np.argmax(z,axis=1)

    # Return an array of y values corresponding to the argmax z value
    if spec_type == "max":    
        y_max = y[z_max]
        # if debug: print("Spectrum arrays (x,y):",x.shape,y_max.shape, sep=" ")
        return x,y_max
    
    if spec_type == "mean":
        # Find the z-weighted mean of the y values along the y-axis of the z array. Consider the z array is 2D and the y array is 1D.
        y_mean = np.sum(y*z,axis=1)/np.sum(z,axis=1)

        # if debug: print("Spectrum arrays (x,y):",x.shape,y_mean.shape, sep=" ")
        return x,y_mean
    
    if spec_type == "top":
        # Find the last value along the y-axis of the z array that is greater than a threshold value
        threshold = 0.25
        z_max = np.max(z,axis=1)
        y_top = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i,j] > z_max[i]*threshold:
                    y_top[i] = y[j]
        if debug: print("Spectrum arrays (x,y):",x.shape,y_top.shape, sep=" ")
        return x,y_top
    
    if spec_type == "bottom":
        threshold = 0.20
        z_max = np.max(z,axis=1)
        y_bottom = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(y)):
                if z[i,j] > z_max[i]*threshold:
                    y_bottom[i] = y[j]
                    break
        if debug: print("Spectrum arrays (x,y):",x.shape,y_bottom.shape, sep=" ")
        return x,y_bottom

def fit_hist1d(x, y, func="polynomial", trimm=0, debug=False):
    '''
    Given a 1D histogram, fit a function to the histogram.
    **VARIABLES:**
    \n ** - x:** x-axis array.
    \n ** - y:** y-axis array.
    '''
    # Remove x values at the beginning and end of the array
    x = x[trimm:-trimm]
    y = y[trimm:-trimm]

    if func == "linear":
        print("Fitting line...")
        def func(x, *coefficients, debug=False):
            return linear(x, coefficients, debug=debug)
        labels = ["Slope","Intercept"]
        initial_guess = np.random.randn(2)
    
    if func == "polynomial":
        print("Fitting polynomial...")
        def func(x, *coefficients, debug=False):
            return polynomial_line(x, coefficients, debug=debug)
        initial_guess = np.random.randn(3)  # Provide an initial guess for the polynomial coefficients
        labels = len(initial_guess)*["coef"]
    
    if func == "exponential":
        print("Fitting exponential...")
        def func(x, *coefficients,debug=False):
            return exp(x, coefficients, debug=debug)
        labels = ["Amplitude","Tau"]
        initial_guess = (1e2,1e4)
    
    if func == "gauss":
        print("Fitting gaussian...")
        def func(x, *coefficients,debug=False):
            return gauss(x, coefficients, debug=debug)
        labels = ["Amplitude","Mean","Sigma"]
        initial_guess = (np.max(y),x[np.argmax(y)],np.std(y))
        # initial_guess = (0,0,0)
        
    # Fitting the polynomial line to the data
    popt, pcov = curve_fit(func, x, y, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    return func,labels,popt,perr

def get_hist2d_fit(x, y, acc, fig, row, col, trimm=5, spec_type="max", func_type="linear", debug=False):
    '''
    Given x and y arrays, generate a 2D histogram and fit a function to the histogram.
    **VARIABLES:**
    \n ** - x:**    x-axis array.
    \n ** - y:**    y-axis array.
    \n ** - acc:**  number of bins.
    \n ** - fig:**  plotly figure.
    \n ** - row:**  row of subplot.
    \n ** - col:**  column of subplot.
    \n ** - func:** function to fit to histogram (exponential, polynomial, etc.).
    '''
    x_array = np.linspace(np.min(x),np.max(x),acc+1)
    y_array = np.linspace(np.min(y),np.max(y),acc+1)
    h, x, y = np.histogram2d(x,y,bins=[x_array,y_array])
    x, y = (x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2

    x_spec, y_spec = spectrum_hist2d(x,y,h,spec_type=spec_type,debug=debug)
    func, labels, popt, perr = fit_hist1d(x_spec,y_spec,trimm=trimm,func=func_type,debug=debug)
    fig.add_trace(go.Heatmap(z=h.T,x=x,y=y,coloraxis="coloraxis"),row=row,col=col)
    fig.add_trace(go.Scatter(x=x_spec,y=y_spec,mode="markers",marker=dict(color="red"),name="Spectrum"),row=row,col=col)
    fig.add_trace(go.Scatter(x=x,y=func(x,*popt),mode="lines",marker=dict(color="red"),name="Fit",error_y=dict(type='data',array=func(x,*perr),visible=True)),row=row,col=col)
    # Make a zoom to the region of interest
    fig.update_layout(xaxis=dict(range=[np.min(x_spec),np.max(x_spec)]),yaxis=dict(range=[np.min(y_spec),np.max(y_spec)]))
    
    if debug:
        for i in range(len(labels)):
            print_colored("Fit parameter %s: %f +/- %f"%(labels[i],popt[i],perr[i]),"INFO")
    return fig, popt, perr

def get_hist1d_fit(x, acc, fig, row, col, trimm=5, func_type="gauss", debug=False):
    '''
    Given an x array, generate a 1D histogram and fit a function to the histogram.
    **VARIABLES:**
    \n** - x:**    x-axis array.
    \n** - acc:**  number of bins.
    \n** - fig:**  plotly figure.
    \n** - row:**  row of subplot.
    \n** - col:**  column of subplot.
    \n** - func:** function to fit to histogram (exponential, polynomial, etc.).
    '''
    if type(acc) == np.ndarray or type(acc) == list:
        x_array = acc
    elif type(acc) == int or type(acc) == float:
        try: x_array = np.linspace(np.min(x),np.max(x),acc+1)
        except ValueError:
            print_colored("ValueError: x array is empty!","WARNING")
            return fig, [], []
    else:
        print_colored("ValueError: acc must be an integer, float, numpy array, or list!","WARNING")
        return fig, [], []
    
    h, x = np.histogram(x,bins=x_array)
    h = h/np.max(h); x = (x[1:]+x[:-1])/2
    fig.add_trace(go.Bar(x=x,y=h,marker=dict(color="grey"),name="Spectrum"),row=row,col=col)
    fig.update_layout(bargap=0)
    
    try:
        func, labels, popt, perr = fit_hist1d(x,h,trimm=trimm,func=func_type,debug=debug)
        # Add text to the plot with the fit parameters
        text = ""
        for i in range(len(labels)):
            text += "%s: %.2f +/- %.2f\n"%(labels[i],popt[i],perr[i])
        fig.add_trace(go.Scatter(x=x,y=func(x,*popt),mode="lines+markers",line=dict(color="red",shape="hvh"),name="Fit",error_y=dict(type='data',array=func(x,*perr),visible=True)),row=row,col=col)

    except:
        print_colored("Fit could not be performed!","WARNING")
        return fig, [], []
    
    if debug:
        for i in range(len(labels)):
            print_colored("Fit parameter %s: %.2f +/- %.2f"%(labels[i],popt[i],perr[i]),"INFO")
    
    return fig, popt, perr