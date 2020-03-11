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

# Number of bins of variable x_data at which parameters like slope, scatter, variance are computed
sampling_size = 25

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

--------------------------------------
Parameters shared by all functions (for Arya's use):
--------------------------------------

df:
    DataFrame containing all properties

xlabel, ylabel, (zlabel):
    labels of data vectors of interest in df. In case of matrix functions, we pass a list of labels into the "ylabels" parameter.

show_data:
    Used in Plot_Fit function to show the datapoints used to make the LLR fit

cutoff:
    Determines cutoff in x-values. Will use whole inputted dataset to make calculations (unless doing a split case)
    but then will only plot parametes for halos with x-value > cutoff

nBootstrap:
    Sets how many bootstrap realizations are made when determining statistical error in parameters

percentile:
    List whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
    Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value for the param

split_label:
    Label of the data vector used to split the data

split_bins:
    Either number of bins (int), or array of bin_edges. If a number is provided the modules will determine the
    bin_edges themselves using the data vector. By default, edges are set so there are equal number of data points in each bin.

mode:
    Sets how the data is split based on the split variable
    If 'Data', then all halos are binned based on the variable df[split_label]
    If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

labels:
    Allows for user-defined labels for x-axis, y-axis, legend labels.
    Currently some modules let you change the actual x/y labels.
    Others only allow you to input what the property is called. Eg. xlabel -> M_{200c}

nbins:
    Available when plotting PDF. Sets the number of the bins the PDF is split into. Can also input an array, in which
    case it will be read as the edges of the bins.

upperlim:
    Available when plotting PDF. Along with use of 'cutoff', it helps construct PDF of residuals in any width bin of x-values

funcs:
    Available when plotting PDF. A dictionary of format {key: function}, where the function will be run on all the residuals in
    every bootstrap realization. Results for median values and 1sigma bounds will be printed and stored in the Output_Data array

verbose:
    controls the verbosity of the model's output.

----------
Outputs:
----------

Every function contains an Output_Data dictionary whose keys are the data vector names (eg. x, y, slope, scatter), and the values
are the computed properties themselves. Any data shown in plots will, and should, be stored in the Output_data dict.
This Output_Data dict will be returned by the function at the end.
'''

def Plot_Fit(df, xlabel, ylabel, xrange = [], show_data = False, GaussianWidth = 0.2, labels = [], ax=None):

    lm = kllr_model()

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    plt.xscale('log')
    plt.yscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    Output_Data = {}

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    if len(xrange) < 2:
        xrange = [np.min(x_data), np.max(x_data)]

    Mask = (x_data > xrange[0]) & (x_data < xrange[1]) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[Mask], y_data[Mask]

    x, y = lm.fit(x_data, y_data, xrange = xrange,
                  nbins = sampling_size, GaussianWidth = GaussianWidth)[0:2]

    # Add black line around regular line to improve visibility
    plt.plot(10**x, 10**y, lw = 6, c = 'k', label = "")
    plt.plot(10**x, 10**y, lw = 3, c = Colors[0])

    Output_Data['x'] = x
    Output_Data['y'] = y

    # Code for bootstrapping the <y | x> values.
    # We don't use it since the uncertainty is very small in our case (not visible actually)

    # x_output, y_output = np.empty([nBootstrap, sampling_size]), np.empty([nBootstrap, sampling_size])

    # for iBoot in range(nBootstrap):
    #
    #     xx, index = subsample(x_data)
    #     yy = y_data[index]
    #
    #     x_output[iBoot,:], y_output[iBoot,:] = LLR_Fit(xx, yy, xrange = (cutoff, np.sort(x_data)[-20]), nbins = sampling_size)[0:2]

    # plt.plot(10**np.median(x_output, axis = 0), 10**np.median(y_output, axis = 0), lw = 6, c = 'k', label = "")
    # plt.plot(10**np.median(x_output, axis = 0), 10**np.median(y_output, axis = 0), lw = 3, c = Colors[0])
    # plt.fill_between(10**np.median(x_output, axis = 0),
    #                  10**np.percentile(y_output, percentile[0], axis = 0), 10**np.percentile(y_output, percentile[1], axis = 0),
    #                  alpha = 0.3, color = Colors[0])

    if show_data:

        Mask = x_data > cutoff

        plt.scatter(10**x_data[Mask], 10**y_data[Mask], s = 30, alpha = 0.3, c = Colors[0], label = "")

    plt.xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + '$', size = fontsize.ylabel)

    return Output_Data, ax

def Plot_Fit_Split(df, xlabel, ylabel, split_label, split_bins = [], xrange = [], show_data = False,
                   mode = 'Data', GaussianWidth = 0.2, labels = [], ax=None):

    lm = kllr_model()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.xscale('log')
    plt.yscale('log')

    # If 3 labels not inserted, default to column names
    if len(labels) < 3:
        labels = [xlabel, ylabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    if len(xrange) < 2:
        xrange = [np.min(x_data), np.max(x_data)]

    Mask = (x_data > xrange[0]) & (x_data < xrange[1]) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]

    # Choose bin edges for binning data
    if isinstance(split_bins, int):
        if mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif mode == 'Residuals':
            split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
        split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)

    # Define dictionary that will contain values that are being plotted
    # First define it to be a dict of dicts whose first level keys are split_bin number
    Output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Loop over bins in split_variable
    for i in range(len(split_bins) - 1):

        # Mask dataset based on raw value or residuals to select only halos in this bin
        if mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        # Run LLR using JUST the subset
        x, y = lm.fit(x_data[split_Mask], y_data[split_Mask],
                      xrange = xrange, GaussianWidth = GaussianWidth)[0:2]

        # Format label depending on Data or Residuals mode
        if mode == 'Data':
            label = r'$' + str(np.round(split_bins[i],2)) + r'<' + labels[2] + '<' + str(np.round(split_bins[i + 1],2)) + '$'
        elif mode == 'Residuals':
            label = r'$' + str(np.round(split_bins[i],2)) + r'< {\rm res}(' + labels[2] + ')<' + str(np.round(split_bins[i + 1],2)) + '$'

        # Add black line first beneath actual line to enhance visibility
        plt.plot(10**x, 10**y, lw = 6, c = 'k', label = "")
        plt.plot(10**x, 10**y, lw = 3, c = Colors[i], label = label)

        # Store data to be outputted later
        Output_Data['Bin' + str(i)]['x'] = x
        Output_Data['Bin' + str(i)]['y'] = y

        if show_data:

            # Select only data above our cutoff
            Mask = (x_data > xrange[0]) & (x_data < xrange[1])

            # Only display data above our cutoff and of halos within the bins in split_data
            plt.scatter(10**x_data[Mask & split_Mask], 10**y_data[Mask & split_Mask],
                        s = 30, alpha = 0.3, c = Colors[i], label = "")

    plt.xlabel(r'$' + labels[0] + '$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + '$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Fit_Params(df, xlabel, ylabel, xrange = [], nBootstrap = 100, GaussianWidth = 0.2,
                    percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig, ax = plt.subplots(2, figsize=(12,10), sharex = True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')

    ax[0].grid()
    ax[1].grid()

    if len(labels) < 2:
        labels = [xlabel, ylabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary to store output values
    Output_Data = {}

    # Load and mask data
    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    if len(xrange) < 2:
        xrange = [np.min(x_data), np.max(x_data)]

    Mask = (x_data > xrange[0]) & (x_data < xrange[1]) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[Mask], y_data[Mask]

    # Generate new arrays to store params in for each Bootstrap realization
    scatter   = np.empty([nBootstrap, sampling_size])
    slope     = np.empty([nBootstrap, sampling_size])
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
                                                                                       nbins  = sampling_size,
                                                                                       GaussianWidth = GaussianWidth)

    ax[0].plot(10**xline, np.mean(slope, axis=0), lw=3, color = Colors[0])
    ax[0].fill_between(10**xline, np.percentile(slope, 16, axis=0), np.percentile(slope, 84, axis=0),
                     alpha=0.4, label=None, color = Colors[0])
    ax[1].plot(10**xline, np.mean(scatter, axis=0)/np.log10(np.e), lw=3, color = Colors[0])
    ax[1].fill_between(10**xline, np.percentile(scatter, 16, axis=0)/np.log10(np.e), np.percentile(scatter, 84, axis=0)/np.log10(np.e),
                         alpha=0.4, label=None, color = Colors[0])

    # Output Data
    Output_Data['x'] = xline

    Output_Data['slope'] = np.median(slope, axis = 0)
    Output_Data['slope+'] = np.percentile(slope, percentile[0], axis = 0)
    Output_Data['slope-'] = np.percentile(slope, percentile[1], axis = 0)

    # Output data for scatter (in ln terms)
    Output_Data['scatter'] = np.median(scatter, axis = 0)/np.log10(np.e)
    Output_Data['scatter+'] = np.percentile(scatter, percentile[0], axis = 0)/np.log10(np.e)
    Output_Data['scatter-'] = np.percentile(scatter, percentile[1], axis = 0)/np.log10(np.e)

    ax[1].set_xlabel(r'$' + labels[0] + '$',           size = fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Fit_Params_Split(df, xlabel, ylabel, split_label, split_bins = [], mode = 'Data',
                          xrange = [], nBootstrap = 100, GaussianWidth = 0.2,
                          percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig, ax = plt.subplots(2, figsize=(12,10), sharex = True)
        fig.subplots_adjust(hspace=0.05)

    ax[0].grid()
    ax[1].grid()

    # Set x_scale to log. Leave y_scale as is.
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    if len(xrange) < 2:
        xrange = [np.min(x_data), np.max(x_data)]

    Mask = (x_data > xrange[0]) & (x_data < xrange[1]) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif mode == 'Residuals':
            split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]

    # Need to compute residuals if mode == 'Residuals' is chosen
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)

    # Define Output_Data variable to store all computed data that is then plotted
    Output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Determine common max halo mass regardless of how we split the data
    Max = np.sort(x_data)[-20]

    for i in range(len(split_bins) - 1):

        if mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif mode == 'Residuals':
            split_Mask = (split_res <= split_bins[i + 1]) & (split_res > split_bins[i])

        scatter   = np.empty([nBootstrap, sampling_size])
        slope     = np.empty([nBootstrap, sampling_size])
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
               slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy,
                                                           xrange = xrange,
                                                           nbins = sampling_size,
                                                           GaussianWidth = GaussianWidth)

        if mode == 'Data':
            label = r'$' + str(np.round(split_bins[i],2)) + '<' + labels[2] + '<' + str(np.round(split_bins[i + 1],2)) + '$'
        elif mode == 'Residuals':
            label = r'$' + str(np.round(split_bins[i],2)) + r'< {\rm res}(' + labels[2] + ')<' + str(np.round(split_bins[i + 1],2)) + '$'

        ax[0].plot(10**xline, np.median(slope, axis=0), lw=3, label = label, color = Colors[i])
        ax[0].fill_between(10**xline, np.percentile(slope, 16, axis=0), np.percentile(slope, 84, axis=0),
                           alpha=0.4, label=None, color = Colors[i])

        # Divide scatter by log10(e) to get it in ln terms (not log10 terms)
        ax[1].plot(10**xline, np.median(scatter, axis=0)/np.log10(np.e), lw=3,
                   label=label, color = Colors[i])
        ax[1].fill_between(10**xline,
                           np.percentile(scatter, percentile[0], axis=0)/np.log10(np.e),
                           np.percentile(scatter, percentile[1], axis=0)/np.log10(np.e),
                           alpha=0.4, label=None, color = Colors[i])

        # Output xvals
        Output_Data['Bin' + str(i)]['x'] = xline

        # Output data for slope
        Output_Data['Bin' + str(i)]['slope']  = np.median(slope, axis = 0)
        Output_Data['Bin' + str(i)]['slope+'] = np.percentile(slope, percentile[0], axis = 0)
        Output_Data['Bin' + str(i)]['slope-'] = np.percentile(slope, percentile[1], axis = 0)

        # Output data for scatter (in ln terms)
        Output_Data['Bin' + str(i)]['scatter']  = np.median(scatter, axis = 0)/np.log10(np.e)
        Output_Data['Bin' + str(i)]['scatter+'] = np.percentile(scatter, percentile[0], axis = 0)/np.log10(np.e)
        Output_Data['Bin' + str(i)]['scatter-'] = np.percentile(scatter, percentile[1], axis = 0)/np.log10(np.e)

    ax[1].set_xlabel(r'$' + labels[0] + '$',           size = fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,(" + labels[1] + ')$', size = fontsize.ylabel)
    ax[1].legend(fontsize=fontsize.legend)

    return Output_Data, ax


def Plot_Correlation(df, xlabel, ylabel, zlabel, GaussianWidth=0.2, cutoff = 13.5,
                     nBootstrap = 100, percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()
    plt.xscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel, zlabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])

    Mask = (x_data > cutoff - 0.5) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))
    x_data, y_data, z_data = x_data[Mask], y_data[Mask], z_data[Mask]

    xline = np.linspace(cutoff, np.sort(x_data)[-20], sampling_size)
    corr = np.zeros([nBootstrap, len(xline)-1])

    Output_Data = {}

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy, zz = x_data, y_data, z_data
        # All other bootstraps have shuffled data
        else:
            xx, index = lm.subsample(x_data)
            yy = y_data[index]
            zz = z_data[index]

        for i in range(len(xline)-1):

            w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[i], sig=GaussianWidth)
            w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[i + 1], sig=GaussianWidth)

            corr[iBoot, i] = lm.calc_correlation_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

    plt.plot(10**((xline[1:] + xline[:-1]) / 2.), np.mean(corr, axis=0), lw=3, color = Colors[0])
    plt.fill_between(10**((xline[1:] + xline[:-1])/ 2.), np.percentile(corr, percentile[0], axis=0), np.percentile(corr, percentile[1], axis=0),
                     alpha=0.4, label=None, color = Colors[0])

    Output_Data['x'] = (xline[1:] + xline[:-1])/ 2.
    Output_Data['correlation'] = np.median(corr, axis = 0)
    Output_Data['correlation+'] = np.percentile(corr, percentile[0], axis = 0)
    Output_Data['correlation-'] = np.percentile(corr, percentile[1], axis = 0)

    plt.axhline(y = 0.0, color = 'k', lw = 3)
    plt.ylim(ymin = -1., ymax = 1.)

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$\rm r\,\, (' + labels[1] + r'\,-\,' + labels[2] + r')$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Correlation_Split(df, xlabel, ylabel, zlabel, split_label, split_bins = [], mode = 'Data',
                           GaussianWidth=0.2, cutoff = 13.5, nBootstrap = 100,
                           percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()
    plt.xscale('log')

    if len(labels) < 4:
        labels = [xlabel, ylabel, zlabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Load and mask data
    x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel]), np.array(df[split_label])

    Mask = (x_data > cutoff) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, z_data, split_data = x_data[Mask], y_data[Mask], z_data[Mask], split_data[Mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif mode == 'Residuals':
            split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth=GaussianWidth)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
        split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth=GaussianWidth)

    # Dictionary for storing values that are plotted
    Output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Determine common max halo mass regardless of how we split the data
    Max = np.sort(x_data)[-20]

    for i in range(len(split_bins) - 1):

        if mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        xline = np.linspace(cutoff, np.sort(x_data[split_Mask])[-20], sampling_size)
        corr  = np.empty([nBootstrap, len(xline)-1])

        if verbose:
            iterations_list = tqdm(range(nBootstrap))
        else:
            iterations_list = range(nBootstrap)

        for iBoot in iterations_list:

            # First bootstrap realization is always just raw data
            if iBoot == 0:
                xx, yy, zz = x_data[split_Mask], y_data[split_Mask], z_data[split_Mask]
            # All other bootstraps have shuffled data
            else:
                xx, index = lm.subsample(x_data[split_Mask])
                yy = y_data[split_Mask][index]
                zz = z_data[split_Mask][index]

            for j in range(len(xline)-1):

                w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[j], sig=GaussianWidth)
                w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[j + 1], sig=GaussianWidth)

                corr[iBoot, j] = lm.calc_correlation_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

        if mode == 'Data':
            label = r'$' + str(np.round(split_bins[i],2)) + '<' + labels[3] + '<' + str(np.round(split_bins[i + 1],2)) + '$'
        elif mode == 'Residuals':
            label = r'$' + str(np.round(split_bins[i],2)) + r'< {\rm res}(' + labels[3] + ')<' + str(np.round(split_bins[i + 1],2)) + '$'

        plt.plot(10**((xline[1:] + xline[:-1])/2.), np.mean(corr, axis=0), lw=3, color = Colors[i], label = label)
        plt.fill_between(10**((xline[1:] + xline[:-1])/2.), np.percentile(corr, percentile[0], axis=0),
                         np.percentile(corr, percentile[1], axis=0), alpha=0.4, label=None, color = Colors[i])

        Output_Data['Bin' + str(i)]['x'] = (xline[1:] + xline[:-1])/2.

        Output_Data['Bin' + str(i)]['correlation'] = np.median(corr, axis = 0)
        Output_Data['Bin' + str(i)]['correlation+'] = np.percentile(corr, percentile[0], axis = 0)
        Output_Data['Bin' + str(i)]['correlation-'] = np.percentile(corr, percentile[1], axis = 0)

    plt.axhline(y = 0.0, color = 'k', lw = 2)
    plt.ylim(ymin = -1, ymax = 1)

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$\rm r\,\, (' + labels[1] + r'\,-\,' + labels[2] + r')$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Covariance(df, xlabel, ylabel, zlabel, cutoff = 13.5, nBootstrap = 100, GaussianWidth=0.2,
                    percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    plt.grid()
    plt.xscale('log')

    if len(labels) < 3:
        labels = [xlabel, ylabel, zlabel]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])

    Mask = (x_data > cutoff - 0.5) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))
    x_data, y_data, z_data = x_data[Mask], y_data[Mask], z_data[Mask]

    xline = np.linspace(cutoff, np.sort(x_data)[-20], sampling_size)
    cov = np.zeros([nBootstrap, len(xline)-1])

    Output_Data = {}

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy, zz = x_data, y_data, z_data
        # All other bootstraps have shuffled data
        else:
            xx, index = lm.subsample(x_data)
            yy = y_data[index]
            zz = z_data[index]

        for i in range(len(xline)-1):

            w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[i],     sig=GaussianWidth)
            w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[i + 1], sig=GaussianWidth)

            cov[iBoot, i] = lm.calc_covariance_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

    plt.plot(10**((xline[1:] + xline[:-1]) / 2.), np.mean(cov, axis=0)/np.log10(np.e)**2, lw=3, color = Colors[0])
    plt.fill_between(10**((xline[1:] + xline[:-1])/ 2.),
                     np.percentile(cov, 16, axis=0)/np.log10(np.e)**2, np.percentile(cov, 84, axis=0)/np.log10(np.e)**2,
                     alpha=0.4, label=None, color = Colors[0])

    Output_Data['x'] = (xline[1:] + xline[:-1])/ 2.
    Output_Data['covariance'] = np.median(cov, axis = 0)/np.log10(np.e)**2
    Output_Data['covariance+'] = np.percentile(cov, percentile[0], axis = 0)/np.log10(np.e)**2
    Output_Data['covariance-'] = np.percentile(cov, percentile[1], axis = 0)/np.log10(np.e)**2

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$\rm cov\,\, (' + labels[1] + r'\,-\,' + labels[2] + r')$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Covariance_Split(df, xlabel, ylabel, zlabel, split_label, split_bins = [], mode = 'Data',
                          cutoff = 13.5, nBootstrap = 100, GaussianWidth=0.2,
                          percentile = [16., 84.], labels = [], verbose=True, ax=None):

    lm = kllr_model()

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()
    plt.xscale('log')

    if len(labels) < 4:
        labels = [xlabel, ylabel, zlabel, split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Load and mask data
    x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel]), np.array(df[split_label])

    Mask = (x_data > cutoff) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, z_data, split_data = x_data[Mask], y_data[Mask], z_data[Mask], split_data[Mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if mode == 'Data':
            split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif mode == 'Residuals':
            split_res  = calculate_residual(x_data, split_data, (cutoff, 16))
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
        split_res  = calculate_residual(x_data, split_data, (cutoff, 16))

    # Dictionary for storing values that are plotted
    Output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Determine common max halo mass regardless of how we split the data
    Max = np.sort(x_data)[-20]

    for i in range(len(split_bins) - 1):

        if mode == 'Data':
            split_Mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (spltit_res > split_bins[i])

        xline = np.linspace(cutoff, np.sort(x_data[split_Mask])[-20], sampling_size)
        cov = np.empty([nBootstrap, len(xline)-1])

        if verbose:
            iterations_list = tqdm(range(nBootstrap))
        else:
            iterations_list = range(nBootstrap)

        for iBoot in iterations_list:

            # First bootstrap realization is always just raw data
            if iBoot == 0:
                xx, yy, zz = x_data[split_Mask], y_data[split_Mask], z_data[split_Mask]
            # All other bootstraps have shuffled data
            else:
                xx, index = lm.subsample(x_data[split_Mask])
                yy = y_data[split_Mask][index]
                zz = z_data[split_Mask][index]

            for j in range(len(xline)-1):
                w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[j],     sig=GaussianWidth)
                w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[j + 1], sig=GaussianWidth)

                cov[iBoot, j] = lm.calc_covariance_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

        if mode == 'Data':
            label = r'$' + str(np.round(split_bins[i],2)) + '<' + labels[3] + '<' + str(np.round(split_bins[i + 1],2)) + '$'
        elif mode == 'Residuals':
            label = r'$' + str(np.round(split_bins[i],2)) + r'< {\rm res}(' + labels[3] + ')<' + str(np.round(split_bins[i + 1],2)) + '$'

        plt.plot(10**((xline[1:] + xline[:-1])/2.), np.mean(cov, axis=0)/np.log10(np.e)**2, lw = 3, color = Colors[i], label = label)
        plt.fill_between(10**((xline[1:] + xline[:-1])/2.), np.percentile(cov, 16, axis=0)/np.log10(np.e)**2,
                         np.percentile(cov, 84, axis=0)/np.log10(np.e)**2, alpha=0.4, label=None, color = Colors[i])

        Output_Data['Bin' + str(i)]['x'] = (xline[1:] + xline[:-1])/2.

        Output_Data['Bin' + str(i)]['covariance']  = np.median(cov, axis = 0)/np.log10(np.e)**2
        Output_Data['Bin' + str(i)]['covariance+'] = np.percentile(cov, percentile[0], axis = 0)/np.log10(np.e)**2
        Output_Data['Bin' + str(i)]['covariance-'] = np.percentile(cov, percentile[1], axis = 0)/np.log10(np.e)**2

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$\rm cov\,\, (' + labels[1] + r'\,-\,' + labels[2] + r')$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Correlation_Matrix(df, xlabel, ylabels, cutoff = 13.5, nBootstrap = 100,
                            GaussianWidth = 0.2, percentile = [16., 84.],
                            labels = [], verbose=True, ax = None):

    lm = kllr_model()

    # size of correlation matrix
    matrix_size = len(ylabels) - 1

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))
        ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    for i in range(matrix_size):
        for j in range(matrix_size):
            ax[i, j].axis('off')

    if len(labels) < (len(ylabels) + 1):
        ylabels.sort()
        labels = [xlabel] + ylabels
        labels = [r'\rm' + item for item in labels]
    else:
        # Sort ylabels alphebetically but make sure we also sort the label list (if provided) in sync
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
            if j <= i:
                continue

            x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])

            Mask = (x_data > cutoff - 0.5) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))

            x_data, y_data, z_data = x_data[Mask], y_data[Mask], z_data[Mask]

            xline = np.linspace(cutoff, np.sort(x_data)[-20], sampling_size)
            corr = np.zeros([nBootstrap, len(xline)-1])

            for iBoot in range(nBootstrap):

                # First bootstrap realization is always just raw data
                if iBoot == 0:
                    xx, yy, zz = x_data, y_data, z_data
                # All other bootstraps have shuffled data
                else:
                    xx, index = lm.subsample(x_data)
                    yy = y_data[index]
                    zz = z_data[index]

                for k in range(len(xline)-1):
                    w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[k], sig=GaussianWidth)
                    w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[k + 1], sig=GaussianWidth)

                    corr[iBoot, k] = lm.calc_correlation_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

            ax[row, col].set_xscale('log')
            ax[row, col].axis('on')
            ax[row, col].plot(10**((xline[1:] + xline[:-1])/2.), np.mean(corr, axis=0), lw=3,
                              color = Colors[0])
            ax[row, col].fill_between(10**((xline[1:] + xline[:-1])/2.), np.percentile(corr, percentile[0], axis=0),
                             np.percentile(corr, percentile[1], axis=0), alpha=0.4, label=None, color = Colors[0])

            ax[row, col].axhline(y = 0.0, color = 'k', lw = 2)
            ax[row, col].set_ylim(ymin = -1, ymax = 1)
            ax[row, col].grid()

            if col == row:
                ax[row, col].text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                                  horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                                  transform=ax[row, col].transAxes)
            if row == col:
                ax[row, col].set_title(r'$' + labels[1 + i] + r'$', size = fontsize.xlabel)

            ax[row, col].tick_params(axis='y', which='major', labelsize=13)

            row += 1

    return ax


def Plot_Correlation_Matrix_Split(df, xlabel, ylabels, split_label, split_bins = [], mode = 'Data',
                                  cutoff = 13.5, nBootstrap = 100, GaussianWidth = 0.2,
                                  percentile = [16., 84.], labels = [], verbose=True, ax = None):

    lm = kllr_model()

    # size of correlation matrix
    matrix_size = len(ylabels) - 1

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))
        ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    # Set all axes off by default. We will turn on only the lower-left-triangle
    for i in range(matrix_size):
        for j in range(matrix_size):
            ax[i,j].axis('off')

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
            if j <= i:
                continue

            x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel]), np.array(df[split_label])

            Mask = (x_data > cutoff) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

            x_data, y_data, z_data, split_data = x_data[Mask], y_data[Mask], z_data[Mask], split_data[Mask]

            # Choose bin edges for binning data
            if (isinstance(split_bins, int)):
                if mode == 'Data':
                    split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
                elif mode == 'Residuals':
                    split_res  = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)
                    split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
            elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
                split_res = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth = GaussianWidth)

            # Determine common max halo mass regardless of how we split the data
            Max = np.sort(x_data)[-20]

            # Normally, we would define a dictionary for output here
            # However, there is too much data here to print out all data shown in a matrix
            # Instead one can obtain correlation plotting data using just the non-matrix version

            for k in range(len(split_bins) - 1):

                if mode == 'Data':
                    split_Mask = (split_data <= split_bins[k + 1]) & (split_data > split_bins[k])
                elif mode == 'Residuals':
                    split_Mask = (split_res < split_bins[k + 1]) & (split_res > split_bins[k])

                xline = np.linspace(cutoff, np.sort(x_data[split_Mask])[-20], sampling_size)
                corr = np.zeros([nBootstrap, len(xline)-1])

                for iBoot in range(nBootstrap):

                    # First bootstrap realization is always just raw data
                    if iBoot == 0:
                        xx, yy, zz = x_data[split_Mask], y_data[split_Mask], z_data[split_Mask]
                    # All other bootstraps have shuffled data
                    else:
                        xx, index = lm.subsample(x_data[split_Mask])
                        yy = y_data[split_Mask][index]
                        zz = z_data[split_Mask][index]

                    for l in range(len(xline)-1):

                        w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[l], sig=GaussianWidth)
                        w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[l + 1], sig=GaussianWidth)

                        corr[iBoot, l] = lm.calc_correlation_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

                if mode == 'Data':
                    label = r'$' + str(np.round(split_bins[k],2)) + '<' + labels[-1] + '<' + str(np.round(split_bins[k + 1],2)) + '$'
                elif mode == 'Residuals':
                    label = r'$' + str(np.round(split_bins[k],2)) + r'< {\rm res}(' + labels[-1] + ')<' + str(np.round(split_bins[k + 1],2)) + '$'

                ax[row, col].set_xscale('log')
                ax[row, col].axis('on')
                ax[row, col].plot(10**((xline[1:] + xline[:-1])/2.), np.mean(corr, axis=0), lw=3, color=Colors[k], label=label)
                ax[row, col].fill_between(10**((xline[1:] + xline[:-1])/2.), np.percentile(corr, percentile[0], axis=0),
                                 np.percentile(corr, percentile[1], axis=0), alpha=0.4, label=None, color=Colors[k])

            ax[row, col].axhline(y = 0.0, color = 'k', lw = 2)
            ax[row, col].set_ylim(ymin = -1, ymax = 1)
            ax[row, col].grid()

            if col == row:
                ax[row, col].text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                                  horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                                  transform=ax[row, col].transAxes)
            if row == col:
                ax[row, col].set_title(r'$' + labels[1 + i] + r'$', size = fontsize.xlabel)

            ax[row, col].tick_params(axis='y', which='major', labelsize=13)

            row += 1

    if matrix_size%2 == 1:
        ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (1.1, 1.3))
    else:
        ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (0.4, 1.9))

    legend = ax[matrix_size//2, matrix_size//2].get_legend()
    for i in range(len(split_bins) - 1):
        legend.legendHandles[i].set_linewidth(2 + 0.5*matrix_size)

    return ax


def Plot_Covariance_Matrix(df, xlabel, ylabels, cutoff = 13.5, nBootstrap = 100, GaussianWidth = 0.2,
                                  percentile = [16., 84.], labels = [], verbose=True, ax = None):

    lm = kllr_model()

    # size of correlation matrix
    matrix_size = len(ylabels)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))
        ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    for i in range(matrix_size):
        for j in range(matrix_size):
            ax[i,j].axis('off')

    if len(labels) < len(ylabels) + 1:
        ylabels.sort()
        labels = [xlabel] + ylabels
        labels = [r'\rm' + item for item in labels]
    else:
        #Sort ylabels alphebetically but make sure we also sort the user-inputted labels list (if provided) in sync
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
            if j < i:
                continue

            x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])

            Mask = (x_data > cutoff - 0.5) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))

            x_data, y_data, z_data = x_data[Mask], y_data[Mask], z_data[Mask]

            xline = np.linspace(cutoff, np.sort(x_data)[-20], sampling_size)
            cov = np.zeros([nBootstrap, len(xline)-1])

            for iBoot in range(nBootstrap):

                # First bootstrap realization is always just raw data
                if iBoot == 0:
                    xx, yy, zz = x_data, y_data, z_data
                # All other bootstraps have shuffled data
                else:
                    xx, index = lm.subsample(x_data)
                    yy = y_data[index]
                    zz = z_data[index]

                for k in range(len(xline)-1):

                    w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[k], sig=GaussianWidth)
                    w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[k + 1], sig=GaussianWidth)

                    cov[iBoot, k] = lm.calc_covariance_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

            ax[row, col].set_xscale('log')
            ax[row, col].axis('on')
            ax[row, col].plot(10**((xline[1:] + xline[:-1])/2.), np.mean(cov, axis=0)/np.log10(np.e)**2, lw=3,
                              color = Colors[0])
            ax[row, col].fill_between(10**((xline[1:] + xline[:-1])/2.),
                                      np.percentile(cov, percentile[0], axis=0)/np.log10(np.e)**2,
                                      np.percentile(cov, percentile[1], axis=0)/np.log10(np.e)**2,
                                      alpha=0.4, label=None, color = Colors[0])

            ax[row, col].grid()

            if col == row:
                ax[row, col].text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                                  horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                                  transform=ax[row, col].transAxes)
            if row == col:
                ax[row, col].set_title(r'$' + labels[1 + i] + r'$', size = fontsize.xlabel)

            ax[row, col].tick_params(axis='y', which='major', labelsize=13)

            row += 1

    return ax


def Plot_Covariance_Matrix_Split(df, xlabel, ylabels, split_label, split_bins = [], mode = 'Data',
                                 cutoff = 13.5, nBootstrap = 100, GaussianWidth = 0.2,
                                 percentile = [16., 84.], labels = [], verbose=True, ax = None):

    lm = kllr_model()

    # size of correlation matrix
    matrix_size = len(ylabels)

    # Hard-encode size of labels on the x and y axis
    plt.rc('xtick',labelsize=22)
    plt.rc('ytick',labelsize=18)

    if ax == None:
        fig = plt.figure(figsize=(5*matrix_size, 5*matrix_size))
        ax = fig.subplots(matrix_size, matrix_size, sharex = True, sharey = True)

    # Set all axes off by default. We will turn on only the lower-left-triangle
    for i in range(matrix_size):
        for j in range(matrix_size):
            ax[i, j].axis('off')

    if len(labels) < len(ylabels) + 2:
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
            if j < i:
                continue

            x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel]), np.array(df[split_label])

            Mask = (x_data > cutoff) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

            x_data, y_data, z_data, split_data = x_data[Mask], y_data[Mask], z_data[Mask], split_data[Mask]

            # Choose bin edges for binning data
            if (isinstance(split_bins, int)):
                if mode == 'Data':
                    split_bins = [np.percentile(split_data, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
                elif mode == 'Residuals':
                    split_res = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth=GaussianWidth)
                    split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
            elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
                split_res = lm.calculate_residual(x_data, split_data, (cutoff, 16), GaussianWidth=GaussianWidth)

            # Determine common max halo mass regardless of how we split the data
            Max = np.sort(x_data)[-20]

            # Normally, we would define a dictionary for output here
            # However, there is too much data here to print out all data shown in a matrix
            # Instead one can obtain correlation plotting data using just the non-matrix version
            for k in range(len(split_bins) - 1):

                if mode == 'Data':
                    split_Mask = (split_data <= split_bins[k + 1]) & (split_data > split_bins[k])
                elif mode == 'Residuals':
                    split_Mask = (split_res < split_bins[k + 1]) & (split_res > split_bins[k])

                xline = np.linspace(cutoff, np.sort(x_data[split_Mask])[-20], sampling_size)
                cov = np.zeros([nBootstrap, len(xline)-1])

                for iBoot in range(nBootstrap):

                    # First bootstrap realization is always just raw data
                    if iBoot == 0:
                        xx, yy, zz = x_data[split_Mask], y_data[split_Mask], z_data[split_Mask]
                    # All other bootstraps have shuffled data
                    else:
                        xx, index = lm.subsample(x_data[split_Mask])
                        yy = y_data[split_Mask][index]
                        zz = z_data[split_Mask][index]

                    for l in range(len(xline)-1):

                        w1 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[l], sig=GaussianWidth)
                        w2 = calculate_weigth(xx, kernel_type='gaussian', mu=xline[l + 1], sig=GaussianWidth)

                        cov[iBoot, l] = lm.calc_covariance_fixed_x(xx, yy, zz, weight=(w1+w2)/2.)

                if mode == 'Data':
                    label = r'$' + str(np.round(split_bins[k],2)) + '<' + labels[-1] + '<' + str(np.round(split_bins[k + 1],2)) + '$'
                elif mode == 'Residuals':
                    label = r'$' + str(np.round(split_bins[k],2)) + r'< {\rm res}(' + labels[-1] + ')<' + str(np.round(split_bins[k + 1],2)) + '$'

                ax[row, col].set_xscale('log')
                ax[row, col].axis('on')
                ax[row, col].plot(10**((xline[1:] + xline[:-1])/2.), np.mean(cov, axis=0)/np.log10(np.e)**2,
                                  lw=3, color = Colors[k], label = label)
                ax[row, col].fill_between(10**((xline[1:] + xline[:-1])/2.), np.percentile(cov, percentile[0], axis=0)/np.log10(np.e)**2,
                                 np.percentile(cov, percentile[1], axis=0)/np.log10(np.e)**2, alpha=0.4, label=None, color = Colors[k])

            ax[row, col].grid()

            if col == row:
                ax[row, col].text(1.02, 0.5, r'$' + labels[1 + j] + r'$', size = fontsize.ylabel,
                                  horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                                  transform=ax[row, col].transAxes)
            if row == col:
                ax[row, col].set_title(r'$' + labels[1 + i] + r'$', size = fontsize.xlabel)

            ax[row, col].tick_params(axis='y', which='major', labelsize=13)

            row += 1

    if matrix_size%2 == 1:
        ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (1.1, 1.3))
    else:
        ax[matrix_size//2, matrix_size//2].legend(prop={'size':8 + 4*matrix_size}, loc = (0.4, 1.9))

    legend = ax[matrix_size//2, matrix_size//2].get_legend()
    for i in range(len(split_bins) - 1):
        legend.legendHandles[i].set_linewidth(2 + 0.5*matrix_size)

    return ax


def Plot_Residual(df, xlabel, ylabel, nbins = 15, cutoff = 13.5, upperlim = np.inf,
                  nBootstrap = 1000, GaussianWidth = 0.2, percentile = [16., 84.],
                  labels = [], funcs = {}, verbose = True, ax = None):

    lm = kllr_model()

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if len(labels) < 2:
        labels = [r'Normalized \,\, Residuals \,\, of \,\, ' + r'\ln(' + ylabel + ')', 'PDF']
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary that will store values to be output
    Output_Data = {}
    results = funcs.keys()

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    Mask = (x_data > cutoff - 0.5) & (x_data < upperlim) & np.invert(np.isinf(y_data) | np.isnan(y_data))

    x_data, y_data = x_data[Mask], y_data[Mask]

    dy = lm.calculate_residual(x_data, y_data, xrange = (cutoff, np.sort(x_data)[-20]),
                               GaussianWidth = GaussianWidth)

    Output_Data['Residuals'] = dy

    PDFs, bins, Output = lm.PDF_generator(dy, nbins, nBootstrap, funcs, density=True, verbose=verbose)

    for r in results:
        min = np.percentile(Output[r], percentile[0])
        mean = np.mean(Output[r])
        max = np.percentile(Output[r], percentile[1])
        print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

        Output_Data[r + '+'] = np.percentile(Output[r], percentile[0])
        Output_Data[r] = np.median(Output[r])
        Output_Data[r + '-'] = np.percentile(Output[r], percentile[1])

    plt.plot(bins, np.mean(PDFs, axis=0), lw = 3, color = Colors[0])
    plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                     alpha=0.4, label=None, color = Colors[0])

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + r'$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax


def Plot_Residual_Split(df, xlabel, ylabel, split_label, split_bins = [], mode = 'Data', nbins = 15, cutoff = 13.5,
                        upperlim = np.inf, nBootstrap = 1000, GaussianWidth = 0.2, percentile = [16., 84.],
                        funcs = {}, labels = [], verbose = True, ax = None):

    lm = kllr_model()

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)

    if len(labels) < 2:
        labels = [r'Normalized \,\, Residuals \,\, of \,\, ' + r'\ln(' + ylabel + ')', 'PDF', split_label]
        # Ensure labels are romanized in tex format if just using label names
        labels = [r'\rm' + item for item in labels]

    # Dictionary that will store values to be output
    Output_Data = {}
    results = funcs.keys()

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    Mask = (x_data > cutoff) & (x_data < upperlim) & np.invert(np.isinf(y_data) | np.isnan(y_data)) & np.invert(np.isinf(split_data) | np.isnan(split_data))

    x_data, y_data, split_data = x_data[Mask], y_data[Mask], split_data[Mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if mode == 'Data':
            # Need temp array so we don't consider values of z for halos with x < cutoff
            split_temp = split_data[x_data > cutoff]
            split_bins = [np.percentile(split_temp, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
        elif mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange = (cutoff, np.sort(x_data)[-20]),
                                              GaussianWidth = GaussianWidth)
            split_bins = [np.percentile(split_res, float(i/split_bins)*100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (mode == 'Residuals'):
        split_res  = lm.calculate_residual(x_data, split_data, xrange = (cutoff, np.sort(x_data)[-20]),
                                           GaussianWidth = GaussianWidth)

    # Define Output_Data variable to store all computed data that is then plotted
    Output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Compute LLR and Residuals using full dataset
    # We do this so the LLR parameters are shared between the different split_bins
    # And that way differences in the PDF are inherent
    # Modulating LLR params for each split_bin would wash away the differences in
    # the PDFs of each split_bin
    dy = lm.calculate_residual(x_data, y_data, xrange = (cutoff, np.sort(x_data)[-20]), GaussianWidth = GaussianWidth)

    # Separately plot the PDF of data in each bin
    for i in range(len(split_bins) - 1):

        if mode == 'Data':
            mask = (x_data > cutoff) & (x_data < np.sort(x_data)[-20])
            split_Mask = (split_data[mask] < split_bins[i + 1]) & (split_data[mask] > split_bins[i])
        elif mode == 'Residuals':
            split_Mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        Output_Data['Bin' + str(i)]['Residuals'] = dy[split_Mask]

        PDFs, bins, Output = lm.PDF_generator(dy[split_Mask], nbins, nBootstrap, funcs, density = True, verbose=verbose)

        for r in results:
            min = np.percentile(Output[r], percentile[0])
            mean = np.mean(Output[r])
            max = np.percentile(Output[r], percentile[1])
            print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

            Output_Data['Bin' + str(i)][r + '+'] = np.percentile(Output[r], percentile[0])
            Output_Data['Bin' + str(i)][r] = np.median(Output[r])
            Output_Data['Bin' + str(i)][r + '-'] = np.percentile(Output[r], percentile[1])

        if mode == 'Data':
            label = r'$' + str(np.round(split_bins[i],2)) + '<' + labels[2] + '<' + str(np.round(split_bins[i + 1],2)) + '$'
        elif mode == 'Residuals':
            label = r'$' + str(np.round(split_bins[i],2)) + r'< {\rm res}(' + labels[2] + ')<' + str(np.round(split_bins[i + 1],2)) + '$'

        plt.plot(bins, np.mean(PDFs, axis=0), lw = 3, color = Colors[i], label = label)
        plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                         alpha=0.4, label=None, color = Colors[i])

    plt.xlabel(r'$' + labels[0] + r'$', size = fontsize.xlabel)
    plt.ylabel(r'$' + labels[1] + r'$', size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    return Output_Data, ax
