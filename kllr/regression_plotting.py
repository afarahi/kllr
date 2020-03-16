import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .regression_model import kllr_model, calculate_weigth

'''
Plotting Params
'''
import matplotlib as mpl
mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.2, 0.8

'''
Initiate Dataset:
    All variables can be accesses and modified as seen fit by user.
    eg. Class.fontsize.xlabel = 0
    eg. Class.Colors = ['red', 'blue', 'aqua']
'''

# Dictionary that allows user to change fontsize of plots whenever necessary
fontsize = pd.Series({
                'title': 25,
                'legend': 18,
                'xlabel': 25,
                'ylabel': 25
                })

# List of colors. First color used by default, but rest used when splitting results by a 3rd variable
Colors = ['#FF7F0E', 'mediumseagreen', 'mediumpurple', 'steelblue']


'''
Plotting functions

-------------------
Disclaimer:
-------------------

(i)   In general, any function that does a split in a 3rd variable only includes data with x_data > cutoff.

(ii)  Functions that do NOT do a split include data from x_data > cutoff - 0.5.

(iii) The choice in (ii) is needed because LLR uses all the data, and so not including data beneath cutoff value
      introduces artifacts at the x-value boundary

(iv)  However, including the x_data > cutoff - 0.5 clause in the split version will mess up how we split the data according to the 3rd variable.
      We split the data into bins in split_variable, where each bin contains an equal number of halos.
      So including the halos below our cutoff will change how our bin-edges are determined and thus affect our results.

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

nbins : int
    Available when plotting PDF. Sets the number of the bins the PDF is split into. Can also input an array, in which
    case it will be read as the edges of the bins.

funcs : dictionary
    Available when plotting PDF. A dictionary of format {key: function}, where the function will be run on all the residuals in
    every bootstrap realization. Results for median values and 1sigma bounds will be printed and stored in the Output_Data array

verbose : boolean
    controls the verbosity of the model's output.

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



## TODO:
    * take care of cutoff
    * give an error : data, ax = Plot_Fit_Split(df, 'M200', 'MStar_BCG100', 'z_form', split_bins=3, split_mode='Residuals')
    * What to do if df labels are int instead of str (eg. 1, 2, 3 etc.)
'''

# constant (set it to np.log(10.0) if you wish to go from dex to fractional error in scatter)
Ln10 = 1.0 # np.log(10.0)

def setup_color(color, split_bins, cmap = None):

    # TODO: through error if the color size does not match color split

    if cmap is None:
        cmap = plt.cm.coolwarm

    if color is None:
        if isinstance(split_bins, int):
            color = cmap(np.linspace(0, 1, split_bins))
        elif isinstance(split_bins, (np.ndarray, list, tuple)):
            color = cmap(np.linspace(0, 1, len(split_bins)-1))

    return color


def Plot_Fit(df, xlabel, ylabel, xrange = None, show_data = False, sampling_size = 25, kernel_type = 'gaussian',
             kernel_width = 0.2, xlog = False, ylog = False, color = None, labels = [], ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    output_Data = {}

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    x, y = lm.fit(x_data, y_data, xrange = xrange, nbins = sampling_size)[0:2]

    if xlog: x = 10**x
    if ylog: y = 10**y

    # Add black line around regular line to improve visibility
    plt.plot(x, y, lw = 6, c = 'k', label = "")
    plt.plot(x, y, lw = 3, c = color)

    output_Data['x'] = x
    output_Data['y'] = y

    # Code for bootstrapping the <y | x> values.
    # We don't use it since the uncertainty is very small in our case (not visible actually)

    if show_data:
        if xlog: x_data = 10 ** x_data
        if ylog: y_data = 10 ** y_data
        plt.scatter(x_data, y_data, s = 30, alpha = 0.3, c = color, label = "")

    plt.xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + '$', size = fontsize.ylabel)

    return output_Data, ax


def Plot_Fit_Split(df, xlabel, ylabel, split_label, split_bins = [], xrange = None, show_data = False,
                   split_mode = 'Data', sampling_size = 25, kernel_type = 'gaussian', kernel_width = 0.2,
                   xlog = False, ylog = False, color = None, labels = [], ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if ax is None:
        ax = plt.figure(figsize=(12, 8))

    color = setup_color(color, split_bins, cmap=None)

    plt.grid()

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')

    # If 3 labels not inserted, default to column names
    if len(labels) < 3:
        labels = [xlabel, ylabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    # Choose bin edges for binning data
    if isinstance(split_bins, int):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)

    # Define dictionary that will contain values that are being plotted
    # First define it to be a dict of dicts whose first level keys are split_bin number
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Loop over bins in split_variable
    for i in range(len(split_bins) - 1):

        # Mask dataset based on raw value or residuals to select only halos in this bin
        if split_mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        # Run LLR using JUST the subset
        x, y = lm.fit(x_data[split_Mask], y_data[split_Mask], xrange = xrange)[0:2]

        # Format label depending on Data or Residuals mode
        if split_mode == 'Data':
            label = r'$%0.2f < %s < %0.2f$'%(split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}(%s) < %0.2f$'%(split_bins[i], labels[2], split_bins[i + 1])

        if xlog: x = 10 ** x
        if ylog: y = 10 ** y

        # Add black line first beneath actual line to enhance visibility
        plt.plot(x, y, lw = 6, c = 'k', label = "")
        plt.plot(x, y, lw = 3, c = color[i], label = label)

        # Store data to be outputted later
        output_Data['Bin' + str(i)]['x'] = x
        output_Data['Bin' + str(i)]['y'] = y

        if show_data:

            if xlog: x_data_tmp = 10 ** x_data
            else: x_data_tmp = x_data
            if ylog: y_data_tmp = 10 ** y_data
            else: y_data_tmp = y_data

            # Select only data above our cutoff
            mask = np.invert(np.isinf(x_data))

            # Only display data above our cutoff and of halos within the bins in split_data
            plt.scatter(x_data_tmp[mask & split_mask], y_data_tmp[mask & split_mask],
                        s = 30, alpha = 0.3, c = color[i], label = "")

    plt.xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + '$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return output_Data, ax


def Plot_Fit_Params(df, xlabel, ylabel, xrange = None, nBootstrap = 100, sampling_size = 25,
                    kernel_type = 'gaussian', kernel_width = 0.2, percentile = [16., 84.],
                    xlog = False, labels = [], color = None, verbose=True, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12,10), sharex = True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    ax[0].grid()
    ax[1].grid()

    if len(labels) < 2:
        labels = [xlabel, ylabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary to store output values
    output_Data = {}

    # Load and mask data
    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    # Generate new arrays to store params in for each Bootstrap realization
    scatter = np.empty([nBootstrap, sampling_size])
    slope = np.empty([nBootstrap, sampling_size])
    intercept = np.empty([nBootstrap, sampling_size])

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy = x_data, y_data
        # All other bootstraps have shuffled data
        else:
            xx, index = lm.subsample(x_data)
            yy = y_data[index]

        # xline is always the same regardless of bootstrap so don't need 2D array for it.
        # yline is not needed for plotting in this module so it's a 'dummy' variable
        xline, yline, intercept[iBoot, :], slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy,
                                                                                       xrange = xrange,
                                                                                       nbins = sampling_size)
    if xlog: xline = 10**xline

    p = ax[0].plot(xline, np.mean(slope, axis=0), lw=3, color = color)
    color = p[0].get_color()
    ax[0].fill_between(xline, np.percentile(slope, 16, axis=0), np.percentile(slope, 84, axis=0),
                     alpha=0.4, label=None, color = color)
    ax[1].plot(xline, np.mean(scatter, axis=0)*Ln10, lw=3, color = color)
    ax[1].fill_between(xline, np.percentile(scatter, 16, axis=0)*Ln10, np.percentile(scatter, 84, axis=0)*Ln10,
                       alpha=0.4, label=None, color = color)

    # Output Data
    output_Data['x'] = xline

    output_Data['slope'] = np.median(slope, axis = 0)
    output_Data['slope+'] = np.percentile(slope, percentile[0], axis = 0)
    output_Data['slope-'] = np.percentile(slope, percentile[1], axis = 0)

    # Output data for scatter (in ln terms)
    output_Data['scatter'] = np.median(scatter, axis = 0)*Ln10
    output_Data['scatter+'] = np.percentile(scatter, percentile[0], axis = 0)*Ln10
    output_Data['scatter-'] = np.percentile(scatter, percentile[1], axis = 0)*Ln10

    ax[1].set_xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,(%s)$"%labels[1], size = fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,(%s)$"%labels[1], size = fontsize.ylabel)

    return output_Data, ax


def Plot_Fit_Params_Split(df, xlabel, ylabel, split_label, split_bins = [], split_mode = 'Data', xrange = None,
                          nBootstrap = 100, sampling_size = 25, kernel_type = 'gaussian', kernel_width = 0.2,
                          xlog = False, percentile = [16., 84.], color = None, labels = [], verbose=True, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12,10), sharex = True)
        fig.subplots_adjust(hspace=0.05)

    color = setup_color(color, split_bins, cmap=None)

    ax[0].grid()
    ax[1].grid()

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]

    # Need to compute residuals if split_mode == 'Residuals' is chosen
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_Mask = (split_res <= split_bins[i + 1]) & (split_res > split_bins[i])

        scatter = np.empty([nBootstrap, sampling_size])
        slope = np.empty([nBootstrap, sampling_size])
        intercept = np.empty([nBootstrap, sampling_size])

        if verbose:
            iterations_list = tqdm(range(nBootstrap))
        else:
            iterations_list = range(nBootstrap)

        for iBoot in iterations_list:

            # First bootstrap realization is always just raw data
            if iBoot == 0:
                xx, yy = x_data[split_Mask], y_data[split_Mask]
            # All other bootstraps have shuffled data
            else:
                xx, index = lm.subsample(x_data[split_Mask])
                yy = y_data[split_Mask][index]

            xline, yline, intercept[iBoot, :], \
               slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy, xrange = xrange, nbins = sampling_size)

        if split_mode == 'Data':
            label = r'$%0.2f < %s < %0.2f$'%(split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}(%s) < %0.2f$'%(split_bins[i], labels[2], split_bins[i + 1])

        if xlog: xline = 10**xline

        ax[0].plot(xline, np.median(slope, axis=0), lw=3, label = label, color = color[i])
        ax[0].fill_between(xline, np.percentile(slope, 16, axis=0), np.percentile(slope, 84, axis=0),
                           alpha=0.4, label=None, color = color[i])

        # Divide scatter by log10(e) to get it in ln terms (not log10 terms)
        ax[1].plot(xline, np.median(scatter, axis=0)*Ln10, lw=3, label=label, color = color[i])
        ax[1].fill_between(xline,
                           np.percentile(scatter, percentile[0], axis=0)*Ln10,
                           np.percentile(scatter, percentile[1], axis=0)*Ln10,
                           alpha=0.4, label=None, color = color[i])

        # Output xvals
        output_Data['Bin' + str(i)]['x'] = xline

        # Output data for slope
        output_Data['Bin' + str(i)]['slope'] = np.median(slope, axis = 0)
        output_Data['Bin' + str(i)]['slope+'] = np.percentile(slope, percentile[0], axis = 0)
        output_Data['Bin' + str(i)]['slope-'] = np.percentile(slope, percentile[1], axis = 0)

        # Output data for scatter (in ln terms)
        output_Data['Bin' + str(i)]['scatter'] = np.median(scatter, axis = 0)*Ln10
        output_Data['Bin' + str(i)]['scatter+'] = np.percentile(scatter, percentile[0], axis = 0)*Ln10
        output_Data['Bin' + str(i)]['scatter-'] = np.percentile(scatter, percentile[1], axis = 0)*Ln10

    ax[1].set_xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].legend(fontsize=fontsize.legend)

    return output_Data, ax


def Plot_Cov_Corr_Matrix(df, xlabel, ylabels, xrange = None, nBootstrap = 100, Output_mode = 'Covariance',
                         sampling_size = 25, kernel_type = 'gaussian', kernel_width = 0.2, percentile = [16., 84.],
                         xlog = False, labels = [], color = None, verbose=True, ax = None):

    lm = kllr_model(kernel_type, kernel_width)

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax is None:
            fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax is None:
            fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=18)

    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    if len(labels) < (len(ylabels) + 1):
        ylabels.sort()
        labels = [xlabel] + ylabels
        labels = [r'\rm' + item for item in labels]
    else:
        # Sort ylabels alphabetically but make sure we also sort the label list (if provided) in sync
        ylabels, temp = zip(*sorted(zip(ylabels, labels[1:])))
        ylabels, labels[1:] = list(ylabels), list(temp)

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

            x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])

            if xrange is None:
                xrange = [np.min(x_data)-0.001, np.max(x_data)+0.001]

            mask = np.invert(np.isinf(x_data)) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))

            x_data, y_data, z_data = x_data[mask], y_data[mask], z_data[mask]

            xline = np.linspace(xrange[0], xrange[1], sampling_size)
            xline = (xline[1:] + xline[:-1]) / 2.

            cov_corr = np.zeros([nBootstrap, len(xline)])

            for iBoot in range(nBootstrap):

                # First bootstrap realization is always just raw data
                if iBoot == 0:
                    xx, yy, zz = x_data, y_data, z_data
                # All other bootstraps have shuffled data
                else:
                    xx, index = lm.subsample(x_data)
                    yy = y_data[index]
                    zz = z_data[index]

                #Dhayaa: Changing len(xline) - 1 to len(xline)
                for k in range(len(xline)):

                    if Output_mode.lower() in ['covariance', 'cov']:
                        cov_corr[iBoot, k] = lm.calc_covariance_fixed_x(xx, yy, zz, xline[k])
                    elif Output_mode.lower() in ['correlation', 'corr']:
                        cov_corr[iBoot, k] = lm.calc_correlation_fixed_x(xx, yy, zz, xline[k])

            if matrix_size > 1: ax_tmp =  ax[row, col]
            else: ax_tmp =  ax

            if xlog: ax_tmp.set_xscale('log')
            ax_tmp.axis('on')

            if xlog: xline = 10 ** (xline)

            p = ax_tmp.plot(xline, np.mean(cov_corr, axis=0), lw=3, color = color)
            color = p[0].get_color()
            ax_tmp.fill_between(xline, np.percentile(cov_corr, percentile[0], axis=0),
                                np.percentile(cov_corr, percentile[1], axis=0), alpha=0.4, label=None, color = color)
            ax_tmp.grid()

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y = 0.0, color = 'k', lw = 2)
                ax_tmp.set_ylim(ymin = -1, ymax = 1)

            if col == row:
                ax_tmp.text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                            horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                            transform=ax_tmp.transAxes)
            if row == col:
                ax_tmp.set_title(r'$%s$'%labels[1 + i], size = fontsize.xlabel)

            ax_tmp.tick_params(axis='y', which='major', labelsize=13)

            row += 1

    return ax


def Plot_Cov_Corr_Matrix_Split(df, xlabel, ylabels, split_label, split_bins = [], Output_mode = 'Covariance',
                               split_mode = 'Data', xrange = None, nBootstrap = 100, sampling_size = 25,
                               kernel_type = 'gaussian', kernel_width = 0.2, xlog = False, percentile = [16., 84.],
                               labels = [], color = None, verbose=True, ax = None):

    lm = kllr_model(kernel_type, kernel_width)

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax == None:
            fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax == None:
            fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=18)

    color = setup_color(color, split_bins, cmap=None)

    # Set all axes off by default. We will turn on only the lower-left-triangle
    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    if len(labels) < (len(ylabels) + 2):
        ylabels.sort()
        labels = [xlabel] + ylabels + [split_label]
        labels = [r'\rm' + item for item in labels]
    else:
        # Sort ylabels alphebetically but make sure we also sort the label list (if provided) in sync
        ylabels, temp = zip(*sorted(zip(ylabels, labels[1:-1])))
        ylabels, labels[1:-1] = list(ylabels), list(temp)

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

            x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]),\
                                                 np.array(df[zlabel]), np.array(df[split_label])

            if xrange is None:
                xrange = [np.min(x_data)-0.001, np.max(x_data)+0.001]

            mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) &\
                   np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

            x_data, y_data, z_data, split_data = x_data[mask], y_data[mask], z_data[mask], split_data[mask]

            # Choose bin edges for binning data
            if (isinstance(split_bins, int)):
                if split_mode == 'Data':
                    split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
                elif split_mode == 'Residuals':
                    split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)
                    split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
            elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
                split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)

            # Normally, we would define a dictionary for output here
            # However, there is too much data here to print out all data shown in a matrix
            # Instead one can obtain correlation plotting data using just the non-matrix version

            for k in range(len(split_bins) - 1):

                if split_mode == 'Data':
                    split_Mask = (split_data <= split_bins[k + 1]) & (split_data > split_bins[k])
                elif split_mode == 'Residuals':
                    split_Mask = (split_res < split_bins[k + 1]) & (split_res > split_bins[k])

                xline = np.linspace(xrange[0], xrange[1], sampling_size)
                xline = (xline[1:] + xline[:-1]) / 2.

                cov_corr = np.zeros([nBootstrap, len(xline)])

                for iBoot in range(nBootstrap):

                    # First bootstrap realization is always just raw data
                    if iBoot == 0:
                        xx, yy, zz = x_data[split_Mask], y_data[split_Mask], z_data[split_Mask]

                    # All other bootstraps have shuffled data
                    else:
                        xx, index = lm.subsample(x_data[split_Mask])
                        yy = y_data[split_Mask][index]
                        zz = z_data[split_Mask][index]

                    for l in range(len(xline)):

                        if Output_mode.lower() in ['covariance', 'cov']:
                            cov_corr[iBoot, l] = lm.calc_covariance_fixed_x(xx, yy, zz, xline[l])
                        elif Output_mode.lower() in ['correlation', 'corr']:
                            cov_corr[iBoot, l] = lm.calc_correlation_fixed_x(xx, yy, zz, xline[l])

                if split_mode == 'Data':
                    label = r'$%0.2f < %s < %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])
                elif split_mode == 'Residuals':
                    label = r'$%0.2f < {\rm res}(%s) < %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])

                if xlog: xline = 10 ** (xline)

                ax_tmp.set_xscale('log')
                ax_tmp.axis('on')
                ax_tmp.plot(xline, np.mean(cov_corr, axis=0), lw=3, color=color[k], label=label)
                ax_tmp.fill_between(xline,
                                    np.percentile(cov_corr, percentile[0], axis=0),
                                    np.percentile(cov_corr, percentile[1], axis=0),
                                    alpha=0.4, label=None, color=color[k])

            ax_tmp.grid()

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y = 0.0, color = 'k', lw = 2)
                ax_tmp.set_ylim(ymin = -1, ymax = 1)

            if col == row:
                ax_tmp.text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                            horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                            transform=ax_tmp.transAxes)
            if row == col:
                ax_tmp.set_title(r'$' + labels[1 + i] + r'$', size = fontsize.xlabel)

            ax_tmp.tick_params(axis='y', which='major', labelsize=13)

            row += 1

    if matrix_size > 1:
        if matrix_size%2 == 1:
            ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (1.1, 1.3))
        else:
            ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (0.4, 1.9))

        legend = ax[matrix_size//2, matrix_size//2].get_legend()
        for i in range(len(split_bins) - 1):
            legend.legendHandles[i].set_linewidth(2 + 0.5*matrix_size)
    else:
        plt.legend(fontsize=fontsize.legend)

    return ax


def Plot_Residual(df, xlabel, ylabel, nbins = 15, xrange = None, nBootstrap = 1000, kernel_type = 'gaussian',
                  kernel_width = 0.2, percentile = [16., 84.], funcs = {}, labels = [],
                  color = None, verbose = True, ax = None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if len(labels) < 2:
        labels = [r'Normalized \,\, Residuals \,\, of \,\, \ln(%s)'%ylabel, 'PDF']
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary that will store values to be output
    output_Data = {}
    results = funcs.keys()

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data) | np.isnan(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    dy = lm.calculate_residual(x_data, y_data, xrange = xrange)

    output_Data['Residuals'] = dy

    PDFs, bins, output = lm.PDF_generator(dy, nbins, nBootstrap, funcs, density=True, verbose=verbose)

    for r in results:
        min = np.percentile(output[r], percentile[0])
        mean = np.mean(output[r])
        max = np.percentile(output[r], percentile[1])
        print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

        output_Data[r + '+'] = np.percentile(output[r], percentile[0])
        output_Data[r] = np.median(output[r])
        output_Data[r + '-'] = np.percentile(output[r], percentile[1])

    p = plt.plot(bins, np.mean(PDFs, axis=0), lw = 3, color = color)
    color = p[0].get_color()
    plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                     alpha=0.4, label=None, color = color)

    plt.xlabel(r'$%s$' % labels[0], size = fontsize.xlabel)
    plt.ylabel(r'$%s$' % labels[1], size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return output_Data, ax


def Plot_Residual_Split(df, xlabel, ylabel, split_label, split_bins = [], split_mode = 'Data', nbins = 15, xrange = None,
                        nBootstrap = 1000, kernel_type = 'gaussian', kernel_width = 0.2, percentile = [16., 84.],
                        labels = [], funcs = {}, color = None, verbose = True, ax = None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    color = setup_color(color, split_bins, cmap=None)

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if len(labels) < 2:
        labels = [r'Normalized \,\, Residuals \,\, of \,\, ' + r'\ln(' + ylabel + ')', 'PDF', split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary that will store values to be output
    results = funcs.keys()

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data) | np.isnan(y_data)) & np.invert(np.isinf(split_data) | np.isnan(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    # Choose bin edges for binning data
    if isinstance(split_bins, int):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data, xrange = xrange)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Compute LLR and Residuals using full dataset
    # We do this so the LLR parameters are shared between the different split_bins
    # And that way differences in the PDF are inherent
    # Modulating LLR params for each split_bin would wash away the differences in
    # the PDFs of each split_bin
    dy = lm.calculate_residual(x_data, y_data, xrange = xrange)

    # Separately plot the PDF of data in each bin
    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_Mask = (split_data < split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        output_Data['Bin' + str(i)]['Residuals'] = dy[split_Mask]

        PDFs, bins, output = lm.PDF_generator(dy[split_Mask], nbins, nBootstrap, funcs,
                                              density = True, verbose = verbose)

        for r in results:
            min = np.percentile(output[r], percentile[0])
            mean = np.mean(output[r])
            max = np.percentile(output[r], percentile[1])
            print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

            output_Data['Bin' + str(i)][r + '+'] = np.percentile(output[r], percentile[0])
            output_Data['Bin' + str(i)][r] = np.median(output[r])
            output_Data['Bin' + str(i)][r + '-'] = np.percentile(output[r], percentile[1])

        if split_mode == 'Data':
            label = r'$%0.2f < %s < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}(%s) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        plt.plot(bins, np.mean(PDFs, axis=0), lw = 3, color = color[i], label = label)
        plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                         alpha=0.4, label=None, color = color[i])

    plt.xlabel(r'$%s$' % labels[0], size = fontsize.xlabel)
    plt.ylabel(r'$%s$' % labels[1], size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return output_Data, ax
