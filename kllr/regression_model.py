"""
Kernel Localized Linear Regression (KLLR) method.

Introduction:
-------------

Linear regression of the simple least-squares variety has been a canonical method used to characterize
the relation between two variables, but its utility is limited by the fact that it reduces full
population statistics down to three numbers: a slope, normalization and variance/standard deviation.
With large empirical or simulated samples we can perform a more sensitive analysis
using a localized linear regression method (see, Farahi et al. 2018 and Anbajagane et al. 2020).
The KLLR method generates estimates of conditional statistics in terms of the local the slope, normalization,
and covariance. Such a method provides a more nuanced description of population statistics appropriate
for the very large samples with non-linear trends.

This code is an implementation of the Kernel Localized Linear Regression (KLLR) method
that performs a localized Linear regression described in Farahi et al. (2018). It employs
bootstrap re-sampling technique to estimate the uncertainties. We also provide a set of visualization
tools so practitioners can seamlessly generate visualization of the model parameters.


Quickstart:
-----------
To start using KLLR, simply use "from KLLR import kllr_model" to
access the primary functions and class. The exact requirements for the inputs are
listed in the docstring of the kllr_model() class further below.
An example for using KLLR looks like this:

    ------------------------------------------------------------------------
    |                                                                      |
    |    from kllr import kllr_model                                       |
    |                                                                      |
    |    lm = kllr_model()                                                 |
    |    xrange, yrange_mean, intercept, slope, scatter =                  |
    |             lm.fit(x, y, [0.0, 1.0], nbins=11, GaussianWidth=0.2)    |
    |                                                                      |
    ------------------------------------------------------------------------

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn import linear_model, mixture

def scatter_cal(x, y, slope, intercept, dof=None, weight=None):
    """
    This function computes the scatter about the mean relation.

    :param x: numpy array
        the vector of independent variable
    :param y: numpy array
        the vector of dependent variable
    :param slope: float
        the slope of the regression model
    :param intercept: float
        the intercept of the regression model
    :param dof: int
        degree of freedom if known otherwise dof = len(x)
    :param weight: numpy array
        the weight vector if None it set to one

    :return: float
        the standard deviation of residuals about the mean relation
    """

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should be one dimensional numpy array,"
            ": len(X.shape) = %i while len(Y.shape) = %i." % (len(x.shape), len(y.shape)))

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[1] = %i while Y.shape[1] = %i." % (x.shape[0], y.shape[0]))

    if dof == None:
        dof = len(x)

    if weight is None:
        sig2 = sum((np.array(y) - (slope * np.array(x) + intercept)) ** 2) / dof
    else:
        sig2 = np.average((np.array(y) - (slope * np.array(x) + intercept)) ** 2, weights=weight)

    return np.sqrt(sig2)


def calculate_weigth(x, kernel_type='gaussian', *args, **kwargs):
    """
    According to the provided kernel, this function computes the weight assigned to each data point.

    :param x: numpy array
        the vector of independent variable
    :param kernel_type: string
        the kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian
    :param **kwargs:
        if kernel_type = 'gaussian' then 'mu' and 'sig' are the mean and width of the gaussian kernels, respectively.
        if kernel_type = 'uniform' then 'xmin' and 'xmax' are the min and max of the uniform kernels, respectively.

    :return: float
        the weight vector
    """

    if len(x.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X. X  should be one dimensional numpy array,"
            ": len(X.shape) = %i."%(len(x.shape)))

    # the gaussian kernel
    def gaussian_kernel(x, mu=0.0, sig=1.0):
        return np.exp(-(x - mu) ** 2 / 2. / sig ** 2)

    # the uniform kernel
    def uniform_kernel(x, xmin=-1.0, xmax=1.0):
        w = np.zeros(len(x))
        w[(x < xmax) * (x > xmin)] = 1.0
        return w

    if kernel_type == 'gaussian':
        w = gaussian_kernel(x, **kwargs)
    elif kernel_type == 'uniform':
        w = uniform_kernel(x, **kwargs)
    else:
        print("Warning : ", kernel_type, "is not a defined filter.")
        print("It assumes w = 1 for every point.")
        w = np.ones(len(x))

    return w


class kllr_model():

    def __init__(self):
        pass

    def linear_regression(self, x, y, weight=None):
        """
        perform a linear regression given a set of weights and return the normalization, slope, and
        scatter about the mean relation.

        :param x: numpy array
            the vector of independent variable
        :param y: numpy array
            the vector of dependent variable
        :param weight: float
            the weight vector

        :return intercept: float
             intercept
        :return slope: float
             slope
        :return sig: float
             scatter about the mean relation
        """

        if weight is None:
            slope, intercept, r_value, p_value = stats.linregress(x, y)[0:4]
        else:
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(x[:, np.newaxis], y, sample_weight=weight)
            slope = regr.coef_[0]
            intercept = regr.intercept_

        sig = scatter_cal(x, y, slope, intercept, weight=weight)
        return intercept, slope, sig

    def subsample(self, x, length=False):
        """
        This function resamples an array and returns the resampled array
        and its indices (if you need to use it as a mask for other arrays)
        :param x: numpy array
            the vector of independent variable
        :param length: string
            the kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

        :return x[resample]: numpy array,
            the resampled vector
        :return Output resample: numpy array,
            the resample index
        """
        x = np.array(x)
        l = length if length else len(x)
        resample = np.floor(np.random.rand(l) * int(len(x))).astype(int)
        return x[resample], resample

    def calc_correlation_fixed_x(self, x, y, z, weight=None):
        """
        computes the covariance between two variables y and z at fixed x.

        :param x: numpy array
            the vector of independent variable
        :param y: numpy array
            the vector of dependent variable
        :param z: numpy array
            the vector of dependent variable
        :param weight: float
            the weight vector

        :return: float
             correlation coefficient
        """

        intercept, slope, sig = self.linear_regression(x, y, weight=weight)
        dy = y - slope * x - intercept

        intercept, slope, sig = self.linear_regression(x, z, weight=weight)
        dz = z - slope * x - intercept

        if weight is None:
            sig = np.cov(dy, dz)
        else:
            sig = np.cov(dy, dz, aweights=weight)

        return sig[1,0] / np.sqrt(sig[0,0]*sig[1,1])

    def calc_covariance_fixed_x(self, x, y, z, weight=None):
        """
        computes the covariance between two variables y and z at fixed x.

        :param x: numpy array
            the vector of independent variable
        :param y: numpy array
            the vector of dependent variable
        :param z: numpy array
            the vector of dependent variable
        :param weight: float
            the weight vector

        :return: float
             covariance
        """

        intercept, slope, sig = self.linear_regression(x, y, weight=weight)
        dy = y - slope * x - intercept

        intercept, slope, sig = self.linear_regression(x, z, weight=weight)
        dz = z - slope * x - intercept

        if weight is None:
            sig = np.cov(dy, dz)
        else:
            sig = np.cov(dy, dz, aweights=weight)

        return sig[1,0]

    def calculate_residual(self, x, y, xrange, nbins=60, GaussianWidth=0.2):
        """
        This function computes the residuals about the mean relation.

        :param x: numpy array
            the vector of independent variable
        :param y: numpy array
            the vector of dependent variable
        :param xrange: list
            the first element is the min and the second element is the max
        :param nbins: int
            the numbers of data points to compute the local regression parameters
        :param GaussianWidth: float
            constant that sets the width of the gaussian kernel

        :return: numpy array
             residuals
        """

        res = np.array([]) # Array to store residuals
        Index = np.array([]) # array to map what residual belongs to what Halo

        # NOTE: The number of sampling points (60) is currently used as default option
        # changing it only changes accuracy a bit (narrower bins means interpolations are more accurate),
        # and also changes computation time
        xline = np.linspace(xrange[0], xrange[1], nbins)

        # Loop over each bin defined by the bin edges above
        for i in range(len(xline) - 1):

            # Compute weight at each edge
            w1 = calculate_weigth(x, kernel_type='gaussian', mu=xline[i],     sig=GaussianWidth)
            w2 = calculate_weigth(x, kernel_type='gaussian', mu=xline[i + 1], sig=GaussianWidth)

            # Compute expected y-value at each bin-edge
            intercept1, slope1, scatter1 = self.linear_regression(x, y, weight=w1)
            yline1 = slope1 * xline[i] + intercept1
            intercept2, slope2, scatter2 = self.linear_regression(x, y, weight=w2)
            yline2 = slope2 * xline[i + 1] + intercept2

            # Compute slope in this bin
            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])

            # Mask to select only halos in this bin
            mask = (x > xline[i]) & (x < xline[i + 1])

            # Interpolate to get scatter at each halo
            std = scatter1 + (scatter2 - scatter1)/(xline[i + 1] - xline[i])*(x[mask] - xline[i])
            # Interpolate expected y-values and Compute residuals
            dy = y[mask] - (yline1 + (yline2 - yline1)/(xline[i + 1] - xline[i])*(x[mask] - xline[i]))
            res = np.concatenate((res, dy/std))

            # Keep track of an index that maps which residual belongs to which halo
            Index = np.concatenate((Index, np.where(mask)[0]))

        # Reshuffle residuals so that res[i] was computed using the halo with values x[i] and y[i]
        res = np.array(res)[np.argsort(Index)]

        return res

    def PDF_generator(self, res, nbins, nBootstrap = 1000, funcs = {}, xrange = (-4, 4), verbose = True, **kwargs):
        """
        Param res: numpy array
            Data to be binned.
        Param nbins: integer
            Number of bins for the PDF
        Param xrange: tuple contaning 2 values
            tuple containing min and max bin values
        Param nBootstrap: integer
            Number of Bootstrap realizations of the PDF
        Param funcs: dictionary
            Dictionary of functions to apply on the Bootstrapped residuals. Format is {'Name': func}.

        return PDFs: numpy array
            numpy array of size (nBootstrap, nbins) containing all realizations of PDFs
        return bins: numpy array
            Central values of the bins of the PDF
        return Output: Dictionary
            Dictionary with format {'name': result}, where result is the output of user-inputted inputted functions acting on residuals.
        """

        if verbose:
            iterations = tqdm(range(nBootstrap))
        else:
            iterations = range(nBootstrap)

        Output = {}

        # Writing a flag for array creation in case user pases an array defining the bin_edges
        # instead of numbers of bins.
        if isinstance(nbins, np.ndarray):
            print("An Array has been passed into nbins param")
            PDFs = np.empty([nBootstrap, len(nbins) - 1])
        else:
            PDFs = np.empty([nBootstrap, nbins])

        # Generate dictionary whose keys are the same keys as the 'func' dictionary
        for function_name in funcs:
            # Choose to use list over np.array because list can handle many types of data
            Output[function_name] = []

        try:
            density = kwargs['density']
        except:
            density = False

        for iBoot in iterations:

            if iBoot == 0:
                residuals = res
                try:
                    w = kwargs['weights']
                except:
                    w = None
            else:
                residuals, index = self.subsample(res)
                try:
                    # If weights exist, reshuffle them according to subsampling
                    w = kwargs['weights'][index]
                except:
                    w = None

            # Compute PDF and store in one row of 2D array
            PDFs[iBoot, :] = np.histogram(residuals, bins=nbins, range=xrange, weights=w, density=density)[0]

            # For each bootstrap, store function output for each function in funcs
            for function_name in funcs:
                Output[function_name].append(funcs[function_name](residuals))

        if isinstance(nbins, np.ndarray):
            # Set array to be the centers of each bin defined by the input array, nbin
            bins = (nbins[1:] + nbins[:-1]) / 2.
        elif isinstance(nbins, (int)):
            # Generate bin_edges
            bins = np.histogram([], bins=nbins, range=xrange)[1]
            # Set array to be the centers of each bin
            bins = (bins[1:] + bins[:-1]) / 2.

        return PDFs, bins, Output

    def fit(self, x, y, xrange, nbins=25, GaussianWidth=0.2):
        """
        This function computes the local regression parameters at the points within xrange.

        :param x: numpy array
            the vector of independent variable
        :param y: numpy array
            the vector of dependent variable
        :param xrange: list
            the first element is the min and the second element is the max
        :param nbins: int
            the numbers of data points to compute the local regression parameters
        :param GaussianWidth: float
            constant that sets the width of the gaussian kernel

        :return: the regression parameters at the local points
             xline: the local points
             yline_exp: the mean value at the local points
             intercept_exp: the intercept at the local points
             slope_exp: the slope at the local points
             scatter_exp: the scatter around mean relation
        """

        # Define x_values to compute regreesion parameters at
        xline = np.linspace(xrange[0], xrange[1], nbins)

        # Generate array to store output from fit
        yline_exp, slope_exp, intercept_exp, scatter_exp = np.zeros(len(xline)), np.zeros(len(xline)), np.zeros(
            len(xline)), np.zeros(len(xline))

        # loop over every sample point
        for i in range(len(xline)):
            # Generate weights at that sample point
            w = calculate_weigth(x, kernel_type='gaussian', mu=xline[i], sig=GaussianWidth)
            # Compute fit params using linear regressions
            intercept_exp[i], slope_exp[i], scatter_exp[i] = self.linear_regression(x, y, weight=w)
            # Generate expected y_value using fit params
            yline_exp[i] = slope_exp[i] * xline[i] + intercept_exp[i]

        return xline, yline_exp, intercept_exp, slope_exp, scatter_exp
