import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from .regression_model import kllr_model, setup_bins


# Plotting parameters
import matplotlib as mpl

mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 20, 20

# Parameters used in this module
# One dictionary to store default values
# Another that user can view/change as necessary
Default_Params = {'default_cmap'   : plt.cm.coolwarm,
                  'title_fontsize' : 25,
                  'legend_fontsize': 22,
                  'xlabel_fontsize': 30,
                  'ylabel_fontsize': 30,
                  'scatter_factor' : 1.0}

Params = Default_Params.copy()

'''
Plotting functions


Parameters
-------------

df : pandas dataframe
    DataFrame containing all properties

xlabel, ylabel(s) : str
    labels of the data vectors of interest in the dataframe.
    In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

show_data : boolean
    Used in Plot_Fit function to show the datapoints used to make the LLR fit.
    Is set to show_data = False, by default

bins : int
    Sets the number of points within x_range that the parameters are sampled at.
    When plotting a PDF it sets the number of bins the PDF is split into.

xrange : list, tuple, np.array
    A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
    By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

nBootstrap : int
    Sets how many bootstrap realizations are made when determining statistical error in parameters.

percentile : list, tuple, np.array
    List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
    Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

split_label : str
    Label of the data vector used to split the data, or condition the data, on a secondary variable.

split_bins : int, list, tuple, numpy array
    Can be either number of bins (int), or array of bin_edges.

    If an int is provided the modules will determine the bin edges themselves using the data vector.
    By default, edges are set so there are equal number of data points in each bin.
    Note that the bin edges in this case will be determed using all data passed into the function. However,
    the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

    If a list is provided then the list elements serve as the bin edges

split_mode : str
    Sets how the data is split/conditioned based on the split variable
    If 'Data', then all halos are binned based on the variable df[split_label]
    If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

labels : list of str
    Allows for user-defined labels for x-axis, y-axis, legend labels.

funcs : dictionary
    Available when plotting PDF. A dictionary of format {key: function}, where the function will be run on all the residuals in
    every bootstrap realization. Results for median values and 1sigma bounds will be printed and stored in the Output_Data array

verbose : boolean
    Controls the verbosity of the model's output.

fast_calc : boolean
    When False, do nothing
    When True , the method only uses data within 3 x kernel_width from the scale mu.
    It speeds up the calculation by removing objects that have exteremly small weight.


Returns
----------

Dictionary
    A dictionary containing all outputs. Is of the form {parameter_name : numpy-array}
    The keys are parameter names (eg. x, y, slope, scatter), and the values
    are the computed properties themselves. Any data shown in plots will, and should, be stored in the Output_data dict.

    In the split case, the dictionary will be a 2D dictionary of form {Bin_id : {parameter_name : array}}, where Bin_id
    represents the bin, determined by split_variable, within which the parameters were computed

Matplotlib.axes
    Axes on which results were plotted

'''

def setup_color(color, split_bins, cmap=None):
    """
    Takes a list of colors, and the number of split_bins and a color map to generate a color map for each split bin.
    if color is None it use cmap to generate a list of colors.

    Parameters
    ----------
    color : list, or None
        a list of matplotlib colors or None

    split_bins : int or list
        number of split bins or the boundary of split bins

    cmap : cmap
        if color=None, It takes cmap and generate a list of matplotlib colors.

    Returns
    -------
    a list of matplotlib colors
    """

    if color is not None:
        if isinstance(split_bins, int):
            if len(color) < split_bins:
                raise ValueError('len(color) is less than split bins while len(color) and'
                                 ' split bins must have the same length.')
        elif isinstance(split_bins, (np.ndarray, list, tuple)):
            if len(color) < len(split_bins) - 1:
                raise ValueError('len(color) is less than len(split_bins)-1 while '
                                 'len(color) must be larger than split bins.')

    if cmap is None:
        cmap = Params['default_cmap']

    if color is None:
        if isinstance(split_bins, int):
            color = cmap(np.linspace(0, 1, split_bins))
        elif isinstance(split_bins, (np.ndarray, list, tuple)):
            color = cmap(np.linspace(0, 1, len(split_bins) - 1))

    return color


def set_params(**kwargs):

    '''
    Change default params used by all functions here.
    If input contains "reset = True", then all Params
    will be set to their default values.
    '''

    for key in kwargs.keys():

        if key in Params.keys():
            Params[key] = kwargs[key]

        elif key.lower() == 'reset':

            if kwargs[key] == True:

                for key in Default_Params.keys():
                    Params[key] = Default_Params[key]

        else:
            print("Param '%s' is invalid. This was no used to update our settings."%key)


def get_params():

    '''
    Returns the Parameter dictionary of this module
    for user to view.
    '''

    return Params


def Plot_Fit_Summary(df, xlabel, ylabel, y_err=None, bins=25, xrange=None, nBootstrap=100,
                     verbose = True, percentile=[16., 84.], kernel_type='gaussian', kernel_width=0.2, fast_calc = False,
                     show_data=False, xlog=False, ylog=False, color=None, labels=None, ax=None):
    '''

    This function visualizes the estimated local fitted parameters (normalization, slope, and scatter).

    Parameters
    -------------

    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.

    show_data : boolean
        Used in Plot_Fit function to show the datapoints used to make the LLR fit.
        Is set to show_data = False, by default

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    lm = kllr_model(kernel_type, kernel_width)

    # Generate figure if none provided
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 19), sharex = True,
                               gridspec_kw = {'height_ratios':[1.75, 1, 1]})

        plt.subplots_adjust(hspace = 0.05)

    [a.grid(True) for a in ax]

    if xlog: ax[2].set_xscale('log')
    if ylog: ax[0].set_yscale('log')

    if labels is None:
        labels = [xlabel, ylabel]

    # Dictionary that will store our
    # final output to be given to
    # the user
    output_Data = {}

    # Load and clean data
    x_data, y_data = df[xlabel].to_numpy(), df[ylabel].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data)
    x_data, y_data = x_data[Mask], y_data[Mask]

    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    # Perform fit
    Fit_Output = lm.fit(x_data, y_data, y_err_data, xrange, bins, nBootstrap, fast_calc, verbose)

    x, y = Fit_Output[0], Fit_Output[1]
    slope, scatter = Fit_Output[3], Fit_Output[4]
    scatter = scatter * Params['scatter_factor']

    # Reshape outputs so code is general even for nBootstrap = 1
    if nBootstrap == 1: y, slope, scatter = y[None, :], slope[None, :], scatter[None, :]

    if xlog: x = 10 ** x
    if ylog: y = 10 ** y

    # Add black line around regular line to improve visibility
    ax[0].plot(x, np.percentile(y, 50, 0), lw=6, color='k', label="")

    p = ax[0].plot(x, np.percentile(y, 50, 0), lw=3, color=color)
    color = p[0].get_color()

    ax[0].fill_between(x, np.percentile(y, percentile[0], 0), np.percentile(y, percentile[1], 0),
                       lw=3, color=color, alpha = 0.4)

    if show_data:

        # Show raw data only with xrange that the fit was computed for

        if xlog:
            Mask = (x_data >= np.min(np.log10(x))) & (x_data <= np.max(np.log10(x)))
        else:
            Mask = (x_data >= np.min(x)) & (x_data <= np.max(x))

        x_data_show, y_data_show = x_data[Mask], y_data[Mask]

        if xlog: x_data_show = 10 ** x_data_show
        if ylog: y_data_show = 10 ** y_data_show
        ax[0].scatter(x_data_show, y_data_show, s=30, alpha=0.3, color=color, label="")

    ax[1].plot(x, np.percentile(slope, 50, 0), lw=3, color=color)
    ax[1].fill_between(x, np.percentile(slope, percentile[0], 0), np.percentile(slope, percentile[1], 0),
                       alpha=0.4, label=None, color=color)

    ax[2].plot(x, np.percentile(scatter, 50, 0) , lw=3, color=color)
    ax[2].fill_between(x, np.percentile(scatter, percentile[0], 0), np.percentile(scatter, percentile[1], 0),
                       alpha=0.4, label=None, color=color)

    ax[0].set_ylabel(labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_ylabel(r"$\beta\,$(%s)" % labels[1],  size=Params['ylabel_fontsize'])
    ax[2].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[2].set_xlabel(labels[0], size=Params['xlabel_fontsize'])

    output_Data['x']  = x

    output_Data['y']  = np.percentile(y, 50, 0)
    output_Data['y-'] = np.percentile(y, percentile[0], 0)
    output_Data['y+'] = np.percentile(y, percentile[1], 0)

    output_Data['slope']  = np.percentile(slope, 50, 0)
    output_Data['slope-'] = np.percentile(slope, percentile[0], 0)
    output_Data['slope+'] = np.percentile(slope, percentile[1], 0)

    output_Data['scatter']  = np.percentile(scatter, 50, 0)
    output_Data['scatter-'] = np.percentile(scatter, percentile[0], 0)
    output_Data['scatter+'] = np.percentile(scatter, percentile[1], 0)

    return output_Data, ax


def Plot_Fit_Summary_Split(df, xlabel, ylabel, split_label, split_bins=[], split_mode = 'Data', y_err=None, bins=25, xrange=None,
                           nBootstrap=100, verbose = True, percentile=[16., 84.], kernel_type='gaussian', kernel_width=0.2, fast_calc = False,
                           show_data=False, xlog=False, ylog=False, color=None, labels=None, cmap = None, ax=None):
    '''

    This function stratifies data on split variable and then visualizes
     estimated local fitted parameters (normalization, slope, and scatter).

    Parameters
    -------------

    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    show_data : boolean
        Used in Plot_Fit function to show the datapoints used to make the LLR fit.
        Is set to show_data = False, by default

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    split_label : str
        Label of the data vector used to split the data, or condition the data, on a secondary variable.

    split_bins : int, list, tuple, numpy array
        Can be either number of bins (int), or array of bin_edges.

        If an int is provided the modules will determine the bin edges themselves using the data vector.
        By default, edges are set so there are equal number of data points in each bin.
        Note that the bin edges in this case will be determined using all data passed into the function. However,
        the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

        If a list is provided then the list elements serve as the bin edges

    split_mode : str
        Sets how the data is split/conditioned based on the split variable
        If 'Data', then all halos are binned based on the variable df[split_label]
        If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    lm    = kllr_model(kernel_type, kernel_width)
    color = setup_color(color, split_bins, cmap)

    # Generate figure if none provided
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 19), sharex = True,
                               gridspec_kw = {'height_ratios':[1.75, 1, 1]})

        plt.subplots_adjust(hspace = 0.05)

    [a.grid(True) for a in ax]

    if xlog: ax[2].set_xscale('log')
    if ylog: ax[0].set_yscale('log')

    # If 3 labels not inserted, default to column names
    if labels is None:
        labels = [xlabel, ylabel, split_label]

    x_data, y_data, split_data = df[xlabel].to_numpy(), df[ylabel].to_numpy(), df[split_label].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data) & clean_vector(split_data)
    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]



    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    # Choose bin edges for binning data
    if isinstance(split_bins, int):

        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

        elif split_mode == 'Residuals':
            split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)

    # Define dictionary that will contain values that are being plotted
    # First define it to be a dict of dicts whose first level keys are split_bin number
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    for i in range(len(split_bins) - 1):

        # Mask dataset based on raw value or residuals to select only halos in this bin
        if split_mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        # Edge case for y_err
        if y_err is None:
            y_err_data_in = None
        else:
            y_err_data_in = y_err_data[split_Mask]

        # Run KLLR using JUST the subset
        Fit_Output = lm.fit(x_data[split_Mask], y_data[split_Mask], y_err_data_in, xrange, bins, nBootstrap, fast_calc, verbose)

        x, y = Fit_Output[0], Fit_Output[1]
        slope, scatter = Fit_Output[3], Fit_Output[4]

        scatter = scatter * Params['scatter_factor']

        # Reshape outputs so code is general even for nBootstrap = 1
        if nBootstrap == 1: y, slope, scatter = y[None, :], slope[None, :], scatter[None, :]

        # Format label depending on Data or Residuals mode
        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        if xlog: x = 10 ** x
        if ylog: y = 10 ** y

        # Add black line around regular line to improve visibility
        ax[0].plot(x, np.percentile(y, 50, 0), lw=6, color='k', label="")

        ax[0].plot(x, np.percentile(y, 50, 0), lw=3, color=color[i], label = label)
        ax[0].fill_between(x, np.percentile(y, percentile[0], 0), np.percentile(y, percentile[1], 0),
                           lw=3, color=color[i], alpha = 0.4)

        if show_data:

            # Show raw data only with xrange that the fit was computed for
            Mask = (x_data >= np.min(x)) & (x_data <= np.max(x))

            x_data_show, y_data_show = x_data[Mask & split_Mask], y_data[Mask & split_Mask]

            if xlog: x_data_show = 10 ** x_data_show
            if ylog: y_data_show = 10 ** y_data_show
            ax[0].scatter(x_data_show, y_data_show, s=30, alpha=0.3, color = color[i], label="")

        ax[1].plot(x, np.percentile(slope, 50, 0), lw=3, color=color[i])
        ax[1].fill_between(x, np.percentile(slope, percentile[0], 0), np.percentile(slope, percentile[1], 0),
                           alpha=0.4, label=None, color=color[i])

        ax[2].plot(x, np.percentile(scatter, 50, 0), lw=3, color=color[i])
        ax[2].fill_between(x, np.percentile(scatter, percentile[0], 0), np.percentile(scatter, percentile[1], 0),
                           alpha=0.4, label=None, color=color[i])

        output_Data['Bin' + str(i)]['x']  = x

        output_Data['Bin' + str(i)]['y']  = np.percentile(y, 50, 0)
        output_Data['Bin' + str(i)]['y-'] = np.percentile(y, percentile[0], 0)
        output_Data['Bin' + str(i)]['y+'] = np.percentile(y, percentile[1], 0)

        output_Data['Bin' + str(i)]['slope']  = np.percentile(slope, 50, 0)
        output_Data['Bin' + str(i)]['slope-'] = np.percentile(slope, percentile[0], 0)
        output_Data['Bin' + str(i)]['slope+'] = np.percentile(slope, percentile[1], 0)

        output_Data['Bin' + str(i)]['scatter']  = np.percentile(scatter, 50, 0)
        output_Data['Bin' + str(i)]['scatter-'] = np.percentile(scatter, percentile[0], 0)
        output_Data['Bin' + str(i)]['scatter+'] = np.percentile(scatter, percentile[1], 0)

    ax[0].set_ylabel(labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_ylabel(r"$\beta\,$(%s)" % labels[1],  size=Params['ylabel_fontsize'])
    ax[2].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[2].set_xlabel(labels[0], size=Params['xlabel_fontsize'])
    ax[0].legend(fontsize=Params['legend_fontsize'])

    return output_Data, ax


def Plot_Higher_Moments(df, xlabel, ylabel, y_err = None, bins=25, xrange=None, nBootstrap=100, verbose = True,
                        kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.], fast_calc = False,
                        xlog=False, labels=None, color=None, ax=None):
    '''
    This function visualizes the third and forth moment of residuals about the best fit curve.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog: ax[1].set_xscale('log')

    [a.grid(True) for a in ax]

    if labels is None:
        labels = [xlabel, ylabel]

    # Dictionary to store output values
    output_Data = {}

    # Load and Mask data
    x_data, y_data = df[xlabel].to_numpy(), df[ylabel].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data)
    x_data, y_data = x_data[Mask], y_data[Mask]

    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    # xline is always the same regardless of bootstrap so don't need 2D array for it.
    # yline, slope, and scatter are not needed for plotting in this module
    function_output = lm.fit(x_data, y_data, y_err_data, xrange, bins, nBootstrap, fast_calc, verbose, True, True)

    x, skew, kurt = function_output[0], function_output[5], function_output[6]

    # Reshape outputs so code is general even for nBootstrap = 1
    if nBootstrap == 1: skew, kurt = skew[None, :], kurt[None, :]

    if xlog: x = 10 ** x

    p = ax[0].plot(x, np.percentile(skew, 50, 0), lw=3, color=color)
    color = p[0].get_color()
    ax[0].fill_between(x, np.percentile(skew, percentile[0], 0), np.percentile(skew, percentile[1], 0),
                     alpha=0.4, label=None, color=color)

    ax[1].plot(x, np.percentile(kurt, 50, 0), lw=3, color=color)
    ax[1].fill_between(x, np.percentile(kurt, percentile[0], 0), np.percentile(kurt, percentile[1], 0),
                     alpha=0.4, label=None, color=color)

    # Output Data
    output_Data['x'] = x

    output_Data['skew']  = np.median(skew, axis=0)
    output_Data['skew-'] = np.percentile(skew, percentile[0], axis=0)
    output_Data['skew+'] = np.percentile(skew, percentile[1], axis=0)

    output_Data['kurt']  = np.median(kurt, axis=0)
    output_Data['kurt-'] = np.percentile(kurt, percentile[0], axis=0)
    output_Data['kurt+'] = np.percentile(kurt, percentile[1], axis=0)

    ax[0].axhline(0, lw = 3, color = 'k', alpha = 0.05)
    ax[1].axhline(3, lw = 3, color = 'k', alpha = 0.05)

    ax[0].set_ylabel(r"$\gamma\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_ylabel(r"$\kappa\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_xlabel(labels[0], size=Params['xlabel_fontsize'])

    return output_Data, ax


def Plot_Higher_Moments_Split(df, xlabel, ylabel, split_label, split_bins=[], split_mode='Data', y_err = None, bins=25,
                              xrange=None, nBootstrap=100, verbose = True, kernel_type='gaussian', kernel_width=0.2, fast_calc = False,
                              xlog=False, percentile=[16., 84.], color=None, labels=None, cmap = None, ax=None):
    '''
    This function stratifies data on split variable and then visualizes
        the third and forth moment of residuals about the best fit curve.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    split_label : str
        Label of the data vector used to split the data, or condition the data, on a secondary variable.

    split_bins : int, list, tuple, numpy array
        Can be either number of bins (int), or array of bin_edges.

        If an int is provided the modules will determine the bin edges themselves using the data vector.
        By default, edges are set so there are equal number of data points in each bin.
        Note that the bin edges in this case will be determined using all data passed into the function. However,
        the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

        If a list is provided then the list elements serve as the bin edges

    split_mode : str

        Sets how the data is split/conditioned based on the split variable
        If 'Data', then all halos are binned based on the variable df[split_label]
        If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    check_attributes(split_bins=split_bins, split_mode=split_mode)

    lm    = kllr_model(kernel_type, kernel_width)
    color = setup_color(color, split_bins, cmap)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog: ax[1].set_xscale('log')

    [a.grid(True) for a in ax]

    if labels is None:
        labels = [xlabel, ylabel, split_label]

    # Dictionary to store output values
    output_Data = {}

    # Load and Mask data
    x_data, y_data, split_data = df[xlabel].to_numpy(), df[ylabel].to_numpy(), df[split_label].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data) & clean_vector(split_data)
    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]

    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

    # Need to compute residuals if split_mode == 'Residuals' is chosen
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_Mask = (split_res <= split_bins[i + 1]) & (split_res > split_bins[i])


        #Edge case for y_err
        if y_err is None:
            y_err_data_in = None
        else:
            y_err_data_in = y_err_data[split_Mask]

        function_output = lm.fit(x_data[split_Mask], y_data[split_Mask], y_err_data_in,
                                 xrange, bins, nBootstrap, fast_calc, verbose, True, True)
        x, skew, kurt = function_output[0], function_output[5], function_output[6]

        # Reshape outputs so code is general even for nBootstrap = 1
        if nBootstrap == 1: skew, kurt = skew[None, :], kurt[None, :]


        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        if xlog: x = 10 ** x

        ax[0].plot(x, np.percentile(skew, 50, 0), lw=3, color=color[i], label = label)
        ax[0].fill_between(x, np.percentile(skew, percentile[0], 0), np.percentile(skew, percentile[1], 0),
                         alpha=0.4, label=None, color=color[i])

        ax[1].plot(x, np.percentile(kurt, 50, 0), lw=3, color=color[i])
        ax[1].fill_between(x, np.percentile(kurt, percentile[0], 0), np.percentile(kurt, percentile[1], 0),
                         alpha=0.4, label=None, color=color[i])

        # Output Data
        output_Data['Bin' + str(i)]['x'] = x

        output_Data['Bin' + str(i)]['skew']  = np.median(skew, axis=0)
        output_Data['Bin' + str(i)]['skew-'] = np.percentile(skew, percentile[0], axis=0)
        output_Data['Bin' + str(i)]['skew+'] = np.percentile(skew, percentile[1], axis=0)

        output_Data['Bin' + str(i)]['kurt']  = np.median(kurt, axis=0)
        output_Data['Bin' + str(i)]['kurt-'] = np.percentile(kurt, percentile[0], axis=0)
        output_Data['Bin' + str(i)]['kurt+'] = np.percentile(kurt, percentile[1], axis=0)

    ax[0].axhline(0, lw = 3, color = 'k', alpha = 0.05)
    ax[1].axhline(3, lw = 3, color = 'k', alpha = 0.05)

    ax[0].set_ylabel(r"$\gamma\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_ylabel(r"$\kappa\,$(%s)" % labels[1], size=Params['ylabel_fontsize'])
    ax[1].set_xlabel(labels[0], size=Params['xlabel_fontsize'])
    ax[0].legend(fontsize=Params['legend_fontsize'])

    return output_Data, ax


def Plot_Cov_Corr_Matrix(df, xlabel, ylabels, y_errs = None, bins=25, xrange=None, nBootstrap=100, verbose = True,
                         Output_mode='Covariance', kernel_type='gaussian', kernel_width=0.2,
                         percentile=[16., 84.], xlog=False, labels=None, color=None, fast_calc = False,
                         ax=None):
    '''
    This function visualizes estimated local correlation coefficient or
    the covariance between a set of variables.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    check_attributes(Output_mode=Output_mode)

    lm = kllr_model(kernel_type, kernel_width)

    #Dictionary to store values
    output_Data = {}

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=True)

    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    if labels is None:
        labels = [xlabel] + ylabels

    col = -1

    if verbose:
        iterations_list = tqdm(ylabels)
    else:
        iterations_list = ylabels

    for ylabel in iterations_list:

        col += 1
        row = col

        for zlabel in ylabels:

            i, j = ylabels.index(ylabel), ylabels.index(zlabel)

            if Output_mode.lower() in ['covariance', 'cov']:
                if j < i:
                    continue

            elif Output_mode.lower() in ['correlation', 'corr']:
                if j <= i:
                    continue

            x_data, y_data, z_data = df[xlabel].to_numpy(), df[ylabel].to_numpy(), df[zlabel].to_numpy()
            Mask = clean_vector(x_data) & clean_vector(y_data) & clean_vector(z_data)
            x_data, y_data, z_data = x_data[Mask], y_data[Mask], z_data[Mask]

            if y_errs is None:
                y_err_data = None
                z_err_data = None
            else:
                if isinstance(y_errs[i], str):
                    y_err_data = df[y_errs[i]].to_numpy()[Mask]
                else:
                    y_err_data = None

                if isinstance(y_errs[j], str):
                    z_err_data = df[y_errs[j]].to_numpy()[Mask]
                else:
                    z_err_data = None

            if Output_mode.lower() in ['covariance', 'cov']:
                x, cov_corr = lm.covariance(x_data, y_data, z_data, y_err_data, z_err_data, xrange, bins, nBootstrap, fast_calc)
                cov_corr    = cov_corr * Params['scatter_factor']**2

            elif Output_mode.lower() in ['correlation', 'corr']:
                x, cov_corr = lm.correlation(x_data, y_data, z_data, y_err_data, z_err_data, xrange, bins, nBootstrap, fast_calc)

            if nBootstrap == 1: cov_corr = cov_corr[None, :]

            output_Data['x'] = x

            if Output_mode.lower() in ['covariance', 'cov']:
                name = 'cov'
            elif Output_mode.lower() in ['correlation', 'corr']:
                name = 'corr'

            output_Data['%s_%s_%s'%(name, ylabel, zlabel)]  = np.percentile(cov_corr, 50, 0)
            output_Data['%s_%s_%s-'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[0], 0)
            output_Data['%s_%s_%s+'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[1], 0)

            if matrix_size > 1:
                ax_tmp = ax[row, col]
            else:
                ax_tmp = ax

            if xlog: ax_tmp.set_xscale('log')
            ax_tmp.axis('on')

            if xlog: x = 10 ** (x)

            p = ax_tmp.plot(x, np.median(cov_corr, axis=0), lw=3, color=color)
            color = p[0].get_color()
            ax_tmp.fill_between(x, np.percentile(cov_corr, percentile[0], axis=0),
                                np.percentile(cov_corr, percentile[1], axis=0), alpha=0.4, label=None, color=color)
            ax_tmp.grid(True)

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y=0.0, color='k', lw=2)
                ax_tmp.set_ylim(ymin=-1.05, ymax=1.05)

            if col == row:

                # Remove any text that exists already
                for text in ax_tmp.texts:
                    text.set_visible(False)

                ax_tmp.text(1.02, 0.5, labels[1 + j], size=Params['ylabel_fontsize'],
                            horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                            transform=ax_tmp.transAxes)
            if row == col:
                ax_tmp.set_title(labels[1 + i], size=Params['xlabel_fontsize'], pad = 15)

            if row == matrix_size - 1:
                ax_tmp.set_xlabel(labels[0], size=Params['xlabel_fontsize'])

            if col == 0:
                if Output_mode.lower() in ['covariance', 'cov']:
                    ax_tmp.set_ylabel('cov', size=Params['xlabel_fontsize'])
                elif Output_mode.lower() in ['correlation', 'corr']:
                    ax_tmp.set_ylabel('r', size=Params['xlabel_fontsize'])

            ax_tmp.tick_params(axis='both', which='major')

            row += 1

    if Output_mode.lower() in ['correlation', 'corr']: plt.subplots_adjust(hspace=0.04, wspace=0.04)
    else: plt.subplots_adjust(hspace=0.04)

    return output_Data, ax


def Plot_Cov_Corr_Matrix_Split(df, xlabel, ylabels, split_label, split_bins=[], y_errs = None, Output_mode='Covariance',
                               split_mode='Data', bins=25, xrange=None, nBootstrap=100, verbose = True, fast_calc = False,
                               kernel_type='gaussian', kernel_width=0.2, xlog=False, percentile=[16., 84.],
                               labels=None, color=None, ax=None):
    '''
    This function stratifies data on split variable and then visualizes
     estimated local correlation coefficient or the covariance between a set of variables.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    split_label : str
        Label of the data vector used to split the data, or condition the data, on a secondary variable.

    split_bins : int, list, tuple, numpy array
        Can be either number of bins (int), or array of bin_edges.

        If an int is provided the modules will determine the bin edges themselves using the data vector.
        By default, edges are set so there are equal number of data points in each bin.
        Note that the bin edges in this case will be determined using all data passed into the function. However,
        the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

        If a list is provided then the list elements serve as the bin edges

    split_mode : str

        Sets how the data is split/conditioned based on the split variable
        If 'Data', then all halos are binned based on the variable df[split_label]
        If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    verbose : boolean
        Controls the verbosity of the model's output.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    check_attributes(split_bins=split_bins, Output_mode=Output_mode, split_mode=split_mode)

    lm    = kllr_model(kernel_type, kernel_width)
    color = setup_color(color, split_bins, cmap=None)

    # Dictionary to store values
    output_Data = {}

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=True)


    # Set all axes off by default. We will turn on only the lower-left-triangle
    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    # if len(labels) < (len(ylabels) + 2):
    if labels is None:
        labels = [xlabel] + ylabels + [split_label]

    # Value to keep track of which column number we're in (leftmost is col = 0)
    # Set it to -1 here so in the first loop it goes to col = 0
    col = -1

    if verbose:
        iterations_list = tqdm(ylabels)
    else:
        iterations_list = ylabels

    for ylabel in iterations_list:

        col += 1
        # Start from the plot on the diagonal
        row = col

        for zlabel in ylabels:

            i, j = ylabels.index(ylabel), ylabels.index(zlabel)

            # Create condition that prevents double-computing the same correlations (eg. corr(x,y) = corr(y,x))
            if Output_mode.lower() in ['covariance', 'cov']:
                if j < i:
                    continue

            elif Output_mode.lower() in ['correlation', 'corr']:
                if j <= i:
                    continue

            if matrix_size > 1:
                ax_tmp = ax[row, col]
            else:
                ax_tmp = ax

            x_data, y_data, z_data, split_data = df[xlabel].to_numpy(), df[ylabel].to_numpy(), df[zlabel].to_numpy(), df[split_label].to_numpy()
            Mask = clean_vector(x_data) & clean_vector(y_data) & clean_vector(z_data)
            x_data, y_data, z_data, split_data = x_data[Mask], y_data[Mask], z_data[Mask], split_data[Mask]

            # Check if y_err is provided and load it
            if y_errs is None:
                y_err_data = None
                z_err_data = None
            else:
                if isinstance(y_errs[i], str):
                    y_err_data = df[y_errs[i]].to_numpy()[Mask]
                else:
                    y_err_data = None

                if isinstance(y_errs[j], str):
                    z_err_data = df[y_errs[j]].to_numpy()[Mask]
                else:
                    z_err_data = None

            # Choose bin edges for binning data
            if (isinstance(split_bins, int)):
                if split_mode == 'Data':
                    split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in
                                  range(0, split_bins + 1)]
                elif split_mode == 'Residuals':
                    split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)
                    split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in
                                  range(0, split_bins + 1)]
            elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
                split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)

            # Define Output_Data variable to store all computed data that is then plotted
            output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

            for k in range(len(split_bins) - 1):

                if split_mode == 'Data':
                    split_Mask = (split_data <= split_bins[k + 1]) & (split_data > split_bins[k])
                elif split_mode == 'Residuals':
                    split_Mask = (split_res <= split_bins[k + 1]) & (split_res > split_bins[k])

                # Edge case for y_err
                if y_err_data is None:
                    y_err_data_in = None
                else:
                    y_err_data_in = y_err_data[split_Mask]

                if z_err_data is None:
                    z_err_data_in = None
                else:
                    z_err_data_in = z_err_data[split_Mask]

                if Output_mode.lower() in ['covariance', 'cov']:
                    x, cov_corr = lm.covariance(x_data[split_Mask], y_data[split_Mask], z_data[split_Mask], y_err_data_in, z_err_data_in, xrange, bins, nBootstrap, fast_calc)
                    cov_corr    = cov_corr * Params['scatter_factor']**2

                elif Output_mode.lower() in ['correlation', 'corr']:
                    x, cov_corr = lm.correlation(x_data[split_Mask], y_data[split_Mask], z_data[split_Mask], y_err_data_in, z_err_data_in, xrange, bins, nBootstrap, fast_calc)

                if nBootstrap == 1: cov_corr = cov_corr[None, :]

                if split_mode == 'Data':
                    label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])
                elif split_mode == 'Residuals':
                    label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])

                if xlog:
                    x = 10 ** (x)
                    ax_tmp.set_xscale('log')

                ax_tmp.axis('on')
                ax_tmp.plot(x, np.percentile(cov_corr, 50, 0), lw=3, color=color[k], label=label)
                ax_tmp.fill_between(x,
                                    np.percentile(cov_corr, percentile[0], 0),
                                    np.percentile(cov_corr, percentile[1], 0),
                                    alpha=0.4, label=None, color=color[k])

                output_Data['Bin' + str(k)]['x'] = x

                if Output_mode.lower() in ['covariance', 'cov']:
                    name = 'cov'
                elif Output_mode.lower() in ['correlation', 'corr']:
                    name = 'corr'

                output_Data['Bin' + str(k)]['%s_%s_%s'%(name, ylabel, zlabel)]  = np.percentile(cov_corr, 50, 0)
                output_Data['Bin' + str(k)]['%s_%s_%s-'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[0], 0)
                output_Data['Bin' + str(k)]['%s_%s_%s+'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[1], 0)

            ax_tmp.grid(True)

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y=0.0, color='k', lw=2)
                ax_tmp.set_ylim(ymin=-1.05, ymax=1.05)

            if col == row:

                #Remove any text that exists already
                for text in ax_tmp.texts:
                    text.set_visible(False)

                ax_tmp.text(1.02, 0.5, labels[1 + j], size=Params['ylabel_fontsize'],
                            horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                            transform=ax_tmp.transAxes)
            if row == col:
                ax_tmp.set_title(labels[1 + i], size=Params['xlabel_fontsize'], pad = 15)

            if row == matrix_size - 1:
                ax_tmp.set_xlabel(labels[0], size=Params['xlabel_fontsize'])

            if col == 0:
                if Output_mode.lower() in ['correlation', 'corr']:
                    ax_tmp.set_ylabel('r', size=Params['xlabel_fontsize'])
                else:
                    ax_tmp.set_ylabel('cov', size=Params['xlabel_fontsize'])

            ax_tmp.tick_params(axis='both', which='major')

            row += 1

    if matrix_size > 1:
        if matrix_size % 2 == 1:
            ax[matrix_size // 2, matrix_size // 2].legend(prop={'size': 8 + 4 * matrix_size}, loc=(1.1, 1.3))
        else:
            ax[matrix_size // 2, matrix_size // 2].legend(prop={'size': 8 + 4 * matrix_size}, loc=(0.4, 1.9))

        legend = ax[matrix_size // 2, matrix_size // 2].get_legend()
        for i in range(len(split_bins) - 1):
            legend.legendHandles[i].set_linewidth(2 + 0.5 * matrix_size)
        if Output_mode.lower() in ['correlation', 'corr']: plt.subplots_adjust(hspace=0.04, wspace=0.04)
        else: plt.subplots_adjust(hspace=0.04)
    else:
        plt.legend(fontsize=Params['legend_fontsize'])

    return output_Data, ax


def Plot_Residual(df, xlabel, ylabel, y_err = None, bins=25, xrange=None, PDFbins = 15, PDFrange=(-4, 4), nBootstrap=100,
                  return_moments = False, kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.], fast_calc = False,
                  labels=None, color=None, ax=None):
    '''
    This function visualizes the histogram of residuals about the best fit curve.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid(True)

    if labels is None:
        labels = [r'$(y - \langle y \rangle)/\sigma_{y},\,\,[y = $ %s $]$' % ylabel, 'PDF']
    else:
        labels = [r'$(y - \langle y \rangle)/\sigma_{y},\,\,[y = $ %s $]$' % labels[0], labels[1]]

    # Dictionary that will store values to be output
    output_Data = {}

    x_data, y_data = df[xlabel].to_numpy(), df[ylabel].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data)
    x_data, y_data = x_data[Mask], y_data[Mask]

    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    dy = lm.residuals(x_data, y_data, y_err, xrange, bins, nBootstrap, fast_calc)

    if nBootstrap == 1: dy = dy[None, :]

    PDFbins = setup_bins(PDFrange, PDFbins, None)
    PDFs = np.empty([nBootstrap, len(PDFbins) - 1])

    for i in range(nBootstrap):
        PDFs[i, :] = np.histogram(dy[i, :], bins = PDFbins, range = PDFrange, density = True)[0]

    if return_moments:

        mean = np.mean(dy, axis = 1)
        sdev = np.std(dy, axis = 1)
        skew = stats.skew(dy, axis = 1)
        kurt = stats.kurtosis(dy, axis = 1, fisher = False)

    PDFbins_plot = (PDFbins[1:] + PDFbins[:-1]) / 2.

    p = plt.plot(PDFbins_plot, np.percentile(PDFs, 50, 0), lw=3, color=color)
    color = p[0].get_color()
    plt.fill_between(PDFbins_plot, np.percentile(PDFs, percentile[0], 0), np.percentile(PDFs, percentile[1], 0),
                     alpha=0.4, label=None, color=color)

    output_Data['Residuals']  = np.percentile(dy, 50, 0)
    output_Data['Residuals-'] = np.percentile(dy, percentile[0], 0)
    output_Data['Residuals+'] = np.percentile(dy, percentile[1], 0)
    output_Data['PDFs']       = PDFs
    output_Data['PDFbins']    = PDFbins_plot

    if return_moments:
        output_Data['mean']  = np.percentile(mean, 50)
        output_Data['mean-'] = np.percentile(mean, percentile[0])
        output_Data['mean+'] = np.percentile(mean, percentile[1])

        output_Data['sdev']  = np.percentile(sdev, 50)
        output_Data['sdev-'] = np.percentile(sdev, percentile[0])
        output_Data['sdev+'] = np.percentile(sdev, percentile[1])

        output_Data['skew']  = np.percentile(skew, 50)
        output_Data['skew-'] = np.percentile(skew, percentile[0])
        output_Data['skew+'] = np.percentile(skew, percentile[1])

        output_Data['kurt']  = np.percentile(kurt, 50)
        output_Data['kurt-'] = np.percentile(kurt, percentile[0])
        output_Data['kurt+'] = np.percentile(kurt, percentile[1])

    plt.xlabel(labels[0], size=Params['xlabel_fontsize'])
    plt.ylabel(labels[1], size=Params['ylabel_fontsize'])

    return output_Data, ax


def Plot_Residual_Split(df, xlabel, ylabel, split_label, split_bins=[], split_mode='Data', y_err = None, bins=25, xrange=None,
                        PDFbins = 15, PDFrange=(-4, 4), nBootstrap=100, return_moments = False,
                        kernel_type='gaussian', kernel_width=0.2, fast_calc = False,
                        percentile=[16., 84.], labels=None, color=None, cmap = None, ax=None):
    '''
    This function stratifies data on split variable and then visualizes
     the histogram of residuals about the best fit curve.

    Parameters
    -------------
    df : pandas dataframe
        DataFrame containing all properties

    xlabel, ylabel(s) : str
        labels of the data vectors of interest in the dataframe.
        In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

    bins : int
        Sets the number of points within x_range that the parameters are sampled at.
        When plotting a PDF it sets the number of bins the PDF is split into.

    xrange : list, tuple, np.array
        A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
        By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

    nBootstrap : int
        Sets how many bootstrap realizations are made when determining statistical error in parameters.

    percentile : list, tuple, np.array
        List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
        Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

    split_label : str
        Label of the data vector used to split the data, or condition the data, on a secondary variable.

    split_bins : int, list, tuple, numpy array
        Can be either number of bins (int), or array of bin_edges.

        If an int is provided the modules will determine the bin edges themselves using the data vector.
        By default, edges are set so there are equal number of data points in each bin.
        Note that the bin edges in this case will be determined using all data passed into the function. However,
        the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

        If a list is provided then the list elements serve as the bin edges

    split_mode : str

        Sets how the data is split/conditioned based on the split variable
        If 'Data', then all halos are binned based on the variable df[split_label]
        If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

    labels : list of str
        Allows for user-defined labels for x-axis, y-axis, legend labels.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float, optional
        If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernels.
        If kernel_type = 'tophat' then 'width' is the width of the tophat kernels.

    fast_calc : boolean
        When False, do nothing
        When True , the method only uses data within 3 x kernel_width from the scale mu.
          It speeds up the calculation by removing objects that have exteremly small weight.


    Returns
    -------
        output_Data: summary of estimated parameters
        ax: matplot object
    '''

    check_attributes(split_bins=split_bins, split_mode=split_mode)

    lm   = kllr_model(kernel_type, kernel_width)

    if ax is None:
        ax = plt.figure(figsize=(12, 8))

    color = setup_color(color, split_bins, cmap)

    plt.grid(True)

    if labels is None:
        labels = [r'$(y - \langle y \rangle)/\sigma_{y},\,\,[y = $ %s $]$' % ylabel, 'PDF', split_label]
    else:
        labels = [r'$(y - \langle y \rangle)/\sigma_{y},\,\,[y = $ %s $]$' % labels[0], labels[1], labels[2]]

    # Dictionary that will store values to be output
    output_Data = {}

    x_data, y_data, split_data = df[xlabel].to_numpy(), df[ylabel].to_numpy(), df[split_label].to_numpy()
    Mask = clean_vector(x_data) & clean_vector(y_data) & clean_vector(split_data)
    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]

    # Check if y_err is provided and load it
    if isinstance(y_err, str):
        y_err_data = df[y_err].to_numpy()[Mask]

    else:
        y_err_data = None

    # Choose bin edges for binning data
    if isinstance(split_bins, int):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

        elif split_mode == 'Residuals':
            split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.residuals(x_data, split_data, xrange=None, bins = bins, nBootstrap = 1)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Compute LLR and Residuals using full dataset
    # We do this so the LLR parameters are shared between the different split_bins
    # And that way differences in the PDF are inherent
    # Modulating LLR params for each split_bin would wash away the differences in
    # the PDFs of each split_bin
    dy = lm.residuals(x_data, y_data, y_err, None, bins, nBootstrap, fast_calc)

    if nBootstrap == 1: dy = dy[None, :]

    # Setup xrange if it is empty
    if xrange is None:
        xrange = [np.min(x_data), np.max(x_data)]

    elif xrange[0] is None:
        xrange[0] = np.min(x_data)

    elif xrange[1] is None:
        xrange[1] = np.max(x_data)

    # Generate Mask so only objects within xrange are used in PDF
    # Need this to make sure dy and split_data are the same length
    xrange_Mask = (x_data <= xrange[1]) & (x_data >= xrange[0])

    # Separately plot the PDF of data in each bin
    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_Mask = (split_data[xrange_Mask] < split_bins[i + 1]) & (split_data[xrange_Mask] > split_bins[i])

        elif split_mode == 'Residuals':
            split_Mask = (split_res[xrange_Mask] < split_bins[i + 1]) & (split_res[xrange_Mask] > split_bins[i])

        PDFbins = setup_bins(PDFrange, PDFbins, None)
        PDFs = np.empty([nBootstrap, len(PDFbins) - 1])

        for j in range(nBootstrap):
            PDFs[j, :] = np.histogram(dy[j, split_Mask], bins = PDFbins, range = PDFrange, density = True)[0]

        if return_moments:

            mean = np.mean(dy, axis = 1)
            sdev = np.std(dy, axis = 1)
            skew = stats.skew(dy, axis = 1)
            kurt = stats.kurtosis(dy, axis = 1, fisher = False)

        PDFbins_plot = (PDFbins[1:] + PDFbins[:-1]) / 2.

        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        plt.plot(PDFbins_plot, np.percentile(PDFs, 50, 0), lw=3, color=color[i], label = label)
        plt.fill_between(PDFbins_plot, np.percentile(PDFs, percentile[0], 0), np.percentile(PDFs, percentile[1], 0),
                         alpha=0.4, label=None, color=color[i])

        output_Data['Bin' + str(i)]['Residuals']  = np.percentile(dy, 50, 0)[split_Mask]
        output_Data['Bin' + str(i)]['Residuals-'] = np.percentile(dy, percentile[0], 0)[split_Mask]
        output_Data['Bin' + str(i)]['Residuals+'] = np.percentile(dy, percentile[1], 0)[split_Mask]
        output_Data['Bin' + str(i)]['PDFs']       = PDFs
        output_Data['Bin' + str(i)]['PDFbins']    = PDFbins_plot

        if return_moments:
            output_Data['Bin' + str(i)]['mean']  = np.percentile(mean, 50)
            output_Data['Bin' + str(i)]['mean-'] = np.percentile(mean, percentile[0])
            output_Data['Bin' + str(i)]['mean+'] = np.percentile(mean, percentile[1])

            output_Data['Bin' + str(i)]['sdev']  = np.percentile(sdev, 50)
            output_Data['Bin' + str(i)]['sdev-'] = np.percentile(sdev, percentile[0])
            output_Data['Bin' + str(i)]['sdev+'] = np.percentile(sdev, percentile[1])

            output_Data['Bin' + str(i)]['skew']  = np.percentile(skew, 50)
            output_Data['Bin' + str(i)]['skew-'] = np.percentile(skew, percentile[0])
            output_Data['Bin' + str(i)]['skew+'] = np.percentile(skew, percentile[1])

            output_Data['Bin' + str(i)]['kurt']  = np.percentile(kurt, 50)
            output_Data['Bin' + str(i)]['kurt-'] = np.percentile(kurt, percentile[0])
            output_Data['Bin' + str(i)]['kurt+'] = np.percentile(kurt, percentile[1])

    plt.xlabel(labels[0], size=Params['xlabel_fontsize'])
    plt.ylabel(labels[1], size=Params['ylabel_fontsize'])
    plt.legend(fontsize=Params['legend_fontsize'])

    return output_Data, ax


def check_attributes(split_bins=10, Output_mode='corr', split_mode='Data'):
    """
    check if the attributes are in correct format.
    """

    if not isinstance(split_bins, int) and not isinstance(split_bins, (np.ndarray, list, tuple)):
        raise TypeError("split_bins must be an integer number or a list of float numbers, "
                        "split_bins is type '%s' "%type(split_bins))
    elif isinstance(split_bins, int) and split_bins < 2:
        raise ValueError('split_bins must be an integer number larger than 1, split_bins is %i'%split_bins)
    elif isinstance(split_bins, (np.ndarray, list, tuple)) and len(split_bins) <= 1:
        raise ValueError('len(split_bins) must be larger than 1, len(split_bins) is %i'%len(split_bins))

    if Output_mode.lower() not in ['correlation', 'corr', 'covariance', 'cov']:
        raise ValueError("Output_mode must be in ['correlation', 'corr', 'covariance', 'cov']. The passed "
                         "Output_mode is `%s`."%Output_mode)

    if split_mode.lower() not in ['residuals', 'data']:
        raise ValueError("split_mode must be in ['Residuals', 'Data']. The passed "
                         "split_mode is `%s`."%split_mode)


def clean_vector(var):
    '''
    Convenience function that simply checks
    the data vector for any values of inf, -inf, or NaN.
    Returns a Mask that is True only for
    "good" values.
    '''

    #Check which entries are problematic
    Mask = np.isinf(var) | np.isneginf(var) | np.isnan(var)

    if len(var.shape) > 1:
        return np.all(np.invert(Mask), axis = 1)
    elif len(var.shape) == 1:
        return np.invert(Mask)
