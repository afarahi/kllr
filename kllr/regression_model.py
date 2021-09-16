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
    |    lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)     |
    |    xrange, yrange_mean, intercept, slope, scatter, skew, kurt =      |
    |             lm.fit(x, y, bins=11)                                   |
    |                                                                      |
    ------------------------------------------------------------------------

"""

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn import linear_model


def scatter(X, y, slopes, intercept, y_err = None, dof=None, weights=None):
    """
    This function computes the weighted scatter about the mean relation.
    If weights= None, then this is the regular scatter.

    Parameters
    ----------
    X : numpy array
        Independent variable data vector. Can have multiple features

    y : numpy array
        Dependent variable data vector.

    slope : numpy array
        1D array of the slopes of the regression model.
        Each entry is the slope of a particular feature.

    intercept : float
        Intercept of the regression model.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weights: numpy array, optional
        Individual weights for each sample. If None then all
        datapoints are weighted equally.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation.

        If y_err is provided, then the output is a corrected standard
        deviation, scatter_true = \sqrt(\sum res^2 - y_err^2), where
        res are the residuals about the mean relation, and the sum is
        implicitly weighted by the input weights, w.

        If y_err is larger than the residuals on average,
        then the sum is negative and scatter_true is not defined.
        In this case we raise a warning and output the value of the sum,
        without taking a square root, i.e. scatter_true^2

    """

    # If X is provided as a 1D array then convert to
    # 2d array with shape (N, 1)
    if len(X.shape) == 1: X = X[:, None]

    if len(X.shape) > 2:
        raise ValueError(
            "Incompatible dimension for X. X should be a two dimensional numpy array,"
            ": len(X.shape) = %i." %len(X.shape))

    if len(y.shape) != 1:
        raise ValueError(
            "Incompatible dimension for Y. Y should be a one dimensional numpy array,"
            ": len(Y.shape) = %i." %len(y.shape))

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[0] = %i while Y.shape[0] = %i." % (X.shape[0], y.shape[0]))

    if isinstance(y_err, (np.ndarray, list, tuple)):

        y_err = np.asarray(y_err)

        if (y_err <= 0).any():
            raise ValueError("Input y_err contains either zeros or negative values.",
                             "It should contain only positive values.")

    # Make sure slopes is an 1D array
    slopes = np.atleast_1d(slopes)

    if len(slopes.shape) > 1:
        raise ValueError(
            "Incompatible dimension for slopes. It should be a one dimensional numpy array,"
            ": len(slopes.shape) = %i." %len(slopes.shape))

    if dof is None:
        dof = len(X) - 1

    if y_err is None:
        y_err = 0
    else:
        weights= weights/y_err

    if weights is None:
        sig2 = sum((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2 - y_err**2) / dof
    else:
        sig2 = np.average((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2 - y_err**2, weights = weights)
        sig2 /= 1 - np.sum(weights**2)/np.sum(weights)**2 #Required factor for getting unbiased estimate

    if (sig2 < 0) & (y_err is 0):

        print("The uncertainty, y_err, is larger than the instrinsic scatter. " + \
              "The corrected variance, var_true = var_obs - y_err^2, is negative.")

        return sig2

    else:
        return np.sqrt(sig2)


def moments(m, X, y, slopes, intercept, y_err = None, dof=None, weights=None):
    """
    This function computes the local moments about the mean relation,
    given some input weights.

    Parameters
    ----------
    m : int, or list, tuple, numpy array of ints
        Either one, or a set of, moments to be computed. Must be integers.

    X : numpy array
        Independent variable data vector. Can have multiple features

    y : numpy array
        Dependent variable data vector.

    slope : numpy array
        1D array of the slopes of the regression model.
        Each entry is the slope of a particular feature.

    intercept : float
        Intercept of the regression model.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None. Currently, using y_err changes
        only the weighting scheme for the moments, and
        does not involve any further corrections.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weights: numpy array, optional
        Individual weights for each sample. If None it assumes a uniform weight.


    Returns
    -------
    float, or numpy array
        The weighted moments of the data. A single float if only one moment
        was requested, and a numpy array if multiple were requested.

    """

    # If X is provided as a 1D array then convert to
    # 2d array with shape (N, 1)
    if len(X.shape) == 1: X = X[:, None]

    if len(X.shape) > 2:
        raise ValueError(
            "Incompatible dimension for X. X should be a two dimensional numpy array,"
            ": len(X.shape) = %i." %len(X.shape))

    if len(y.shape) != 1:
        raise ValueError(
            "Incompatible dimension for Y. Y should be a one dimensional numpy array,"
            ": len(Y.shape) = %i." %len(Y.shape))

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[0] = %i while Y.shape[0] = %i." % (X.shape[0], y.shape[0]))

    # Make sure slopes is an 1D array
    slopes = np.atleast_1d(slopes)

    if len(slopes.shape) > 1:
        raise ValueError(
            "Incompatible dimension for slopes. It should be a one dimensional numpy array,"
            ": len(slopes.shape) = %i." %len(slopes.shape))

    if isinstance(y_err, (np.ndarray, list, tuple)):

        y_err = np.asarray(y_err)

        if (y_err <= 0).any():
            raise ValueError("Input y_err contains either zeros or negative values. " + \
                             "It should contain only positive values.")

    elif y_err is not None:

        weights= weights/y_err

    if dof is None:
        dof = len(X)

    m      = np.atleast_1d(m).astype(int)
    output = np.zeros(m.size)

    residuals = np.array(y) - (np.dot(X, slopes) + intercept)
    for i in range(output.size):

        if weights is None:
            output[i] = np.sum(residuals**m[i]) / dof

        else:
            output[i] = np.average(residuals**m[i], weights=weights)

    if output.size == 1:
        return output[0]

    else:
        return output


def skewness(X, y, slopes, intercept, y_err = None, dof=None, weights=None):
    """
    This function computes the weighted skewness about the mean relation.
    If weights= None, then this is the regular skewness.

    Parameters
    ----------
    X : numpy array
        Independent variable data vector. Can have multiple features

    y : numpy array
        Dependent variable data vector.

    slope : numpy array
        1D array of the slopes of the regression model.
        Each entry is the slope of a particular feature.

    intercept : float
        Intercept of the regression model.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weights: numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The weighted skewness of the sample.
        It is just the standard skewness if Weights = None.

    """

    m2, m3 = moments([2, 3], X, y, slopes, intercept, y_err, dof, weights)

    skew = m3/m2**(3/2)

    return skew


def kurtosis(X, y, slopes, intercept, y_err = None, dof=None, weights=None):
    """
    This function computes the weighted kurtosis about the mean relation.
    If weights= None, then this is the regular skewness.

    Parameters
    ----------
    X : numpy array
        Independent variable data vector. Can have multiple features

    y : numpy array
        Dependent variable data vector.

    slope : numpy array
        1D array of the slopes of the regression model.
        Each entry is the slope of a particular feature.

    intercept : float
        Intercept of the regression model.

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weights: numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation

    """

    m2, m4 = moments([2, 4], X, y, slopes, intercept, y_err, dof, weights)

    kurt = m4/m2**2

    return kurt


def calculate_weights(x, kernel_type='gaussian', mu=0, width=0.2):
    """
    According to the provided kernel, this function computes the weights assigned to each data point.

    Parameters
    ----------
    x : numpy array
        A one dimensional data vector.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    mu, width : float, optional
        If kernel_type = 'gaussian' then 'mu' and 'width' are the mean and width of the gaussian kernels, respectively.
        If kernel_type = 'tophat' then 'mu' and 'width' are the mean and width of the tophat kernels, respectively.

    Returns
    -------
    float
        the weights vector
    """

    if len(x.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X. X  should be one dimensional numpy array,"
            ": len(X.shape) = %i." % (len(x.shape)))

    # the gaussian kernel
    def gaussian_kernel(x, mu=0.0, width=1.0):
        return 1/np.sqrt(2*np.pi*width**2)*np.exp(-(x - mu) ** 2 / 2. / width ** 2)

    # the tophat kernel
    def tophat_kernel(x, mu=0.0, width=1.0):
        w = np.zeros(len(x))
        w[np.abs(x - mu) < width/2] = 1.0
        return w

    if kernel_type == 'gaussian':
        w = gaussian_kernel(x, mu=mu, width=width)
    elif kernel_type == 'tophat':
        w = tophat_kernel(x, mu=mu, width=width)
    else:
        print("Warning : ", kernel_type, "is not a defined filter.")
        print("It assumes w = 1 for every point.")
        w = np.ones(len(x))

    return w


def setup_bins(xrange, bins, x):
    """
    Convenience function that generates sample points for regression

    Parameters
    ----------
    xrange : list, array
        2-element array [min, max] of the range the regression is performed over.
        If min and/or max is set to None, then we use the vector x to determine it.

    bins : int, or list, tuple, array
        If "int", then we use xrange and data vector to compute the sampling points
        If list, or array, then the input is used as the sampling points.

    x : numpy array
        Data vector of the independent variable in the regression.
        If xrange == None, then we use this data vector to set the range of the regression

    Returns
    -------
    numpy array
        sampling points of the regression
    """

    if isinstance(bins, (np.ndarray, list, tuple)):

        return np.asarray(bins)

    elif isinstance(bins, int):

        if xrange is None:
            xrange = (np.min(x), np.max(x))

        elif xrange[0] is None:
            xrange[0] = np.min(x)

        elif xrange[1] is None:
            xrange[1] = np.max(x)

        xline = np.linspace(xrange[0], xrange[1], bins, endpoint=True)

        return xline


def setup_kernel_width(kernel_width, default_width, bins_size):
    """
    Convenience function that sets up the kernel_width values

    Parameters
    ----------
    kernel_width : int/float, or a list/array

    default_width : int, or list, tuple, array
        Default value to assign kernel_width, if input is kernel_width == None

    bins_size : int
        Number of sampling points in the regression problem

    Returns
    -------
    numpy array
        kernel_widths for each sampling point in the regression problem
    """

    if kernel_width is None:

        return np.ones(bins_size)*default_width

    elif isinstance(kernel_width, (float, int)):

        return np.ones(bins_size)*kernel_width

    elif isinstance(kernel_width, (list, np.ndarray)):

        if kernel_width.size != bins_size:
            raise ValueError("Size mismatch. kernel_width is size %d, but we have %d sampling points.\n "%(kernel_width.size, bins_size))

        else:

            return kernel_width


class kllr_model():
    """
    A class used to represent a KLLR model and perform the fit. It is supported by additional functions that allows
    to compute the conditional properties such as residuals about the mean relation,
    the correlation coefficient, and the covariance.

    Attributes
    ----------
    kernel_type : string
        The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

    kernel_width : float
         If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
         If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.

    Methods
    -------
    linear_regression(x, y, y_err = None, weights = None)
        perform a linear regression give a set of weights

    correlation(self, data_x, data_y, data_z, x, y_err = None, z_err = None, fast_calc = False, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    covariance(x, y, xrange = None, bins = 60, y_err = None, z_err = None, fast_calc = False, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    residuals(x, y, y_err = None, fast_calc = False, xrange = None, bins = 60,  kernel_type = None, kernel_width = None)
        compute residuals about the mean relation i.e., res = y - <y|X>

    fit(x, y, y_err = None, xrange = None, fast_calc = False, bins = 25, kernel_type = None, kernel_width = None)
        fit a kernel localized linear relation to (x, y) pairs, i.e. <y | x> = a(y) x + b(y)

    """

    def __init__(self, kernel_type='gaussian', kernel_width=0.2):
        """
        Parameters
        ----------
        kernel_type : string, optional
            the kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel. The default is Gaussian

        kernel_width : float, optional
            if kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            if kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
        """

        self.kernel_type  = kernel_type
        self.kernel_width = kernel_width

    def linear_regression(self, X, y, y_err = None, weights=None, compute_skewness = False, compute_kurtosis = False):
        """
        This function perform a linear regression given a set of weights
        and return the normalization, slope, and scatter about the mean relation.


        Parameters
        ----------
        X : numpy array
            Independent variable data vector.
            Can input multiple features.

        y : numpy array
            Dependent variable data vector.

        y_err : numpy array, optional
            Uncertainty on dependent variable, y.
            Must contain only non-zero positive values.
            Default is None.

        weights: float, optional
            Individual weights for each sample. If none it assumes a uniform weight.
            If X has multiple features then weights are applied to only using the
            first feature (or first column)

        compute_skewness : boolean, optional
            If compute_skewness == True, the weighted skewness
            is computed and returned in the output

        compute_kurtosis : boolean, optional
            If compute_kurtosis == True, the weighted kurtosis
            is computed and returned in the output

        Returns
        -------
        float
             intercept

        numpy-array
             Array of slopes, with the size
             given by the number of features. If X is
             1D then the slope output is still a 1D array
             with size 1.

        float
             scatter about the mean relation

        float or None, optional
             skewness about the mean relation.
             Present only if compute_skewness = True and
             is None if compute_skewness = False

        float or None, optional
             kurtosis about the mean relation
             Present only if compute_kurtosis = True and
             is None if compute_kurtosis = False
        """

        # if X is not 2D then raise error
        if len(X.shape) > 2:
            raise ValueError("Incompatible dimension for X."
                             "X must be a numpy array with atmost two dimensions.")

        elif len(X.shape) == 1:

            X = X[:, None] # convert 1D to 2D array


        if len(y.shape) != 1:
            raise ValueError(
                "Incompatible dimension for Y. Y should be a one dimensional numpy array,"
                ": len(Y.shape) = %i." %len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
                ": X.shape[0] = %i while Y.shape[0] = %i." % (X.shape[0], y.shape[0]))

        # If y_err is an array/list, check that all values are positive
        if isinstance(y_err, (np.ndarray, list, tuple)):

            y_err = np.asarray(y_err)

            if (y_err <= 0).any():
                raise ValueError("Input y_err contains either zeros or negative values. " + \
                                 "It should contain only positive values.")

        regr = linear_model.LinearRegression()
        # Train the model using the training sets

        if y_err is None:
            regr.fit(X, y, sample_weight=weights)

        elif (weights is not None) and (y_err is not None):
            regr.fit(X, y, sample_weight=weights/y_err)

        elif (weights is None) and (y_err is not None):
            regr.fit(X, y, sample_weight=1/y_err)

        slopes = regr.coef_
        intercept = regr.intercept_

        sig = scatter(X, y, slopes, intercept, y_err, weights=weights)

        skew, kurt = None, None #Set some default values

        if compute_skewness: skew = skewness(X, y, slopes, intercept, weights=weights)
        if compute_kurtosis: kurt = kurtosis(X, y, slopes, intercept, weights=weights)

        return intercept, slopes, sig, skew, kurt

    def fit(self, X, y, y_err = None, xrange = None, bins = 25, nBootstrap = 100,
            fast_calc = False, verbose = False, compute_skewness = False, compute_kurtosis = False,
            kernel_type = None, kernel_width = None):
        """
        This function computes the local regression parameters at the points within xrange.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        y_err : numpy array, optional
            Uncertainty on dependent variable, y.
            Must contain only non-zero positive values.
            Default is None.

        xrange : list, optional
            The first element is the min and the second element is the max,
            If None, it sets xrange to [min(x), max(x)]

        bins : int, optional
            The numbers of data points to compute the local regression parameters

        compute_skewness : boolean, optional
            If compute_skewness == True, the weighted skewness
            is computed and returned in the output

        compute_kurtosis : boolean, optional
            If compute_kurtosis == True, the weighted kurtosis
            is computed and returned in the output

        kernel_type : string, optional
            The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
            If None it uses the pre-specified `kernel_width`.

        Returns
        -------
        numpy-array
            The local points.

        numpy-array
            The mean value at the local points

        numpy-array
            The intercept at the local points

        numpy-array
            The slope at the local points

        numpy-array
            The scatter around mean relation

        numpy-array, optional
             skewness about the mean relation.
             Present only if compute_skewness = True and
             array contains only None elements
             if compute_skewness = False

        numpy-array, optional
             kurtosis about the mean relation
             Present only if compute_kurtosis = True and
             array contains only None elements
             if compute_kurtosis = False
        """

        if len(X.shape) == 1: X = X[:, None] #Make sure X is atleast 2D

        # Define x_values to compute regression parameters at
        xline = setup_bins(xrange, bins, X[:, 0])
        kernel_width = setup_kernel_width(kernel_width, self.kernel_width, xline.size)

        if kernel_type is None: kernel_type = self.kernel_type

        # Generate array to store output from fit
        slopes = np.zeros(shape=(nBootstrap, xline.size, X.shape[1]))
        yline, intercept, scatter, skew, kurt = [np.zeros([nBootstrap, xline.size]) for i in range(5)]

        # If X has multiple features, we cannot compute an expectation value <y | X>
        # that is just a line (it would be in a N-D plane instead). So set yline = None then.
        if X.shape[1] > 1:
            yline = None

        if verbose: iterator = tqdm(range(xline.size))
        else: iterator = range(xline.size)

        # loop over every sample point
        for i in iterator:

            if fast_calc:

                Mask = (X[:, 0] > xline[i] - kernel_width[i]*3) & (X[:, 0] < xline[i] + kernel_width[i]*3)
                X_small, y_small = X[Mask, :], y[Mask]

                if y_err is None:
                    y_err_small = None
                elif isinstance(y_err, np.ndarray):
                    y_err_small = y_err[Mask]

                if X_small.size == 0:

                    raise ValueError("Attempting regression using 0 objects at x = %0.2f. To correct this\n"%xline[i] + \
                                     "you can (i) set fast_calc = False, (ii) increase kernel width, or;\n" + \
                                     "(iii) perform KLLR over an xrange that excludes x = %0.2f"%xline[i])

            else:

                X_small, y_small, y_err_small = X, y, y_err

            # Generate weights at sample point
            w = calculate_weights(X_small[:, 0], kernel_type = kernel_type, mu = xline[i], width = kernel_width[i])

            for j in range(nBootstrap):

                #First "bootstrap" is always using unsampled data
                if j == 0:
                    rand_ind = np.ones(y_small.size).astype(bool)
                else:
                    rand_ind = np.random.randint(0, y_small.size, y_small.size)

                #Edge case handling I:
                #If y_err is a None, then we can't index it
                if y_err_small is None:
                    y_err_small_in = None
                elif isinstance(y_err_small, np.ndarray):
                    y_err_small_in = y_err_small[rand_ind]

                # Compute fit params using linear regressions
                output = self.linear_regression(X_small[rand_ind], y_small[rand_ind],
                                                y_err_small_in, w[rand_ind],
                                                compute_skewness, compute_kurtosis)

                intercept[j, i] = output[0]
                slopes[j, i]    = output[1]
                scatter[j, i]   = output[2]
                skew[j, i]      = output[3]
                kurt[j, i]      = output[4]

                if X.shape[1] == 1:
                    # Generate expected y_value using fit params
                    yline[j, i] = slopes[j, i, 0] * xline[i] + intercept[j, i]

        if nBootstrap == 1:

            yline     = np.squeeze(yline, 0)
            intercept = np.squeeze(intercept, 0)
            slopes    = np.squeeze(slopes, 0)
            scatter   = np.squeeze(scatter, 0)
            skew      = np.squeeze(skew, 0)
            kurt      = np.squeeze(kurt, 0)

        if X.shape[1] == 1:

            slopes = np.squeeze(slopes, -1)

        return xline, yline, intercept, slopes, scatter, skew, kurt

    def correlation(self, X, y, z, y_err = None, z_err = None, xrange = None, bins = 25, nBootstrap = 100,
                    fast_calc = False, verbose = False, kernel_type=None, kernel_width=None):

        """
        This function computes the correlatio between two variables y and z,
        conditioned on all the properties in data vector X.

        Parameters
        ----------
        X : numpy array
            Independent variable data vector. Can contain multiple features.

        y : numpy array
            Dependent variable data vector. Must be a one dimensional data vector.

        z : numpy array
            Dependent variable data vector. Must be a one dimensional data vector.

        fast_calc : boolean
            When False, do nothing
            When True , the method only uses data within 3 x kernel_width from the scale mu.
             It speeds up the calculation by removing objects that have extremely small weight.

        y_err, z_err : numpy array, optional
            Uncertainty on dependent variable, y and z.
            Must contain only non-zero positive values.
            Default is None.

        verbose : boolean
            Controls the verbosity of the model's output.

        xrange : float
            Value of the conditional parameter. It computes the covariance at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        float
             Covariance.
        """

        if len(X.shape) == 1: X = X[:, None] #Make sure X is atleast 2D

        xline = setup_bins(xrange, bins, X[:, 0])
        kernel_width = setup_kernel_width(kernel_width, self.kernel_width, xline.size)

        if kernel_type is None:
            kernel_type = self.kernel_type

        correlation = np.zeros([nBootstrap, xline.size])

        if verbose: iterator = tqdm(range(xline.size))
        else: iterator = range(xline.size)

        # loop over every sample point
        for i in iterator:

            if fast_calc:

                Mask = (X[:, 0] > xline[i] - kernel_width[i]*3) & (X[:, 0] < xline[i] + kernel_width[i]*3)
                X_small, y_small, z_small = X[Mask, :], y[Mask], z[Mask]

                if y_err is None:
                    y_err_small = None
                elif isinstance(y_err, np.ndarray):
                    y_err_small = y_err[Mask]

                if z_err is None:
                    z_err_small = None
                elif isinstance(z_err, np.ndarray):
                    z_err_small = z_err[Mask]

                if X_small.size == 0:

                    raise ValueError("Attempting regression using 0 objects at x = %0.2f. To correct this\n"%xline[i] + \
                                     "you can (i) set fast_calc = False, (ii) increase kernel width, or;\n" + \
                                     "(iii) perform KLLR over an xrange that excludes x = %0.2f"%xline[i])

            else:

                X_small, y_small, z_small, y_err_small, z_err_small = X, y, z, y_err, z_err

            # Generate weights at sample point
            w = calculate_weights(X_small[:, 0], kernel_type = kernel_type, mu=xline[i], width=kernel_width[i])

            for j in range(nBootstrap):

                # First "bootstrap" is always using unsampled data
                if j == 0:
                    rand_ind = np.ones(y_small.size).astype(bool)
                else:
                    rand_ind = np.random.randint(0, y_small.size, y_small.size)

                # Store the shuffled variables so you don't have to
                # compute the shuffle multiple times

                X_small_rand = X_small[rand_ind]
                y_small_rand = y_small[rand_ind]
                z_small_rand = z_small[rand_ind]
                w_rand       = w[rand_ind]

                # Edge case handling I:
                # If y_err is a None, then we can't index it
                if y_err_small is None:
                    y_err_small_in = None
                elif isinstance(y_err_small, np.ndarray):
                    y_err_small_in = y_err_small[rand_ind]

                if z_err_small is None:
                    z_err_small_in = None
                elif isinstance(z_err_small, np.ndarray):
                    z_err_small_in = z_err_small[rand_ind]

                # Compute fit params using linear regressions
                intercept, slope = self.linear_regression(X_small_rand, y_small_rand, y_err_small_in, weights = w_rand)[:2]
                dy = y_small_rand - (intercept + np.dot(X_small_rand, slope))

                intercept, slope = self.linear_regression(X_small_rand, z_small_rand, z_err_small_in, weights = w_rand)[:2]
                dz = z_small_rand - (intercept + np.dot(X_small_rand, slope))

                cov = np.cov(dy, dz, aweights = w_rand)
                correlation[j, i] = cov[1, 0]/np.sqrt(cov[0,0] * cov[1,1])

        if nBootstrap == 1: correlation  = np.squeeze(correlation, 0)

        return xline, correlation

    def covariance(self, X, y, z, y_err = None, z_err = None, xrange = None, bins = 25, nBootstrap = 100,
                   fast_calc = False, verbose = False, kernel_type=None, kernel_width=None):

        """
        This function computes the covariance between two variables y and z,
        conditioned on all the properties in data vector X.

        Parameters
        ----------
        X : numpy array
            Independent variable data vector. Can contain multiple features.

        y : numpy array
            Dependent variable data vector. Must be a one dimensional data vector.

        z : numpy array
            Dependent variable data vector. Must be a one dimensional data vector.

        y_err, z_err : numpy array, optional
            Uncertainty on dependent variable, y and z.
            Must contain only non-zero positive values.
            Default is None.

        xrange : float
            Value of the conditional parameter. It computes the covariance at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
            If None it uses the pre-specified `kernel_width`

        fast_calc : boolean
            When False, do nothing
            When True , the method only uses data within 3 x kernel_width from the scale mu.
             It speeds up the calculation by removing objects that have extremely small weight.


        Returns
        -------
        float
             Covariance.
        """

        if len(X.shape) == 1: X = X[:, None] #Make sure X is atleast 2D

        xline = setup_bins(xrange, bins, X[:, 0])
        kernel_width = setup_kernel_width(kernel_width, self.kernel_width, xline.size)

        if kernel_type is None:
            kernel_type = self.kernel_type

        covariance = np.zeros([nBootstrap, xline.size])

        if verbose: iterator = tqdm(range(xline.size))
        else: iterator = range(xline.size)

        # loop over every sample point
        for i in iterator:

            if fast_calc:

                Mask = (X[:, 0] > xline[i] - kernel_width[i]*3) & (X[:, 0] < xline[i] + kernel_width[i]*3)
                X_small, y_small, z_small = X[Mask, :], y[Mask], z[Mask]

                if y_err is None:
                    y_err_small = None
                elif isinstance(y_err, np.ndarray):
                    y_err_small = y_err[Mask]

                if z_err is None:
                    z_err_small = None
                elif isinstance(z_err, np.ndarray):
                    z_err_small = z_err[Mask]

                if X_small.size == 0:

                    raise ValueError("Attempting regression using 0 objects at x = %0.2f. To correct this\n"%xline[i] + \
                                     "you can (i) set fast_calc = False, (ii) increase kernel width, or;\n" + \
                                     "(iii) perform KLLR over an xrange that excludes x = %0.2f"%xline[i])

            else:

                X_small, y_small, z_small, y_err_small, z_err_small = X, y, z, y_err, z_err

            # Generate weights at sample point
            w = calculate_weights(X_small[:, 0], kernel_type = kernel_type, mu=xline[i], width=kernel_width[i])

            for j in range(nBootstrap):

                #First "bootstrap" is always using unsampled data
                if j == 0:
                    rand_ind = np.ones(y_small.size).astype(bool)
                else:
                    rand_ind = np.random.randint(0, y_small.size, y_small.size)

                # Store the shuffled variables so you don't have to
                # compute the shuffle multiple times

                X_small_rand = X_small[rand_ind]
                y_small_rand = y_small[rand_ind]
                z_small_rand = z_small[rand_ind]
                w_rand       = w[rand_ind]

                #Edge case handling I:
                #If y_err is a None, then we can't index it
                if y_err_small is None:
                    y_err_small_in = None
                elif isinstance(y_err_small, np.ndarray):
                    y_err_small_in = y_err_small[rand_ind]

                if z_err_small is None:
                    z_err_small_in = None
                elif isinstance(z_err_small, np.ndarray):
                    z_err_small_in = z_err_small[rand_ind]

                # Compute fit params using linear regressions
                intercept, slope = self.linear_regression(X_small_rand, y_small_rand, y_err_small_in, weights = w_rand)[:2]
                dy = y_small_rand - (intercept + np.dot(X_small_rand, slope))

                intercept, slope = self.linear_regression(X_small_rand, z_small_rand, z_err_small_in, weights = w_rand)[:2]
                dz = z_small_rand - (intercept + np.dot(X_small_rand, slope))

                cov = np.cov(dy, dz, aweights = w_rand)
                covariance[j, i] = cov[1, 0]

        if nBootstrap == 1: covariance  = np.squeeze(covariance, 0)

        return xline, covariance

    def residuals(self, X, y, y_err = None, xrange=None, bins=25, nBootstrap = 100,
                  fast_calc = False, verbose = False, kernel_type=None, kernel_width=None):
        """
        This function computes the residuals about the mean relation, i.e. res = y - <y | x>.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        y_err : numpy array, optional
            Uncertainty on dependent variable, y.
            Must contain only non-zero positive values.
            Default is None.

        xrange : list, optional
            The range of regression. The first element is the min and the second element is the max.
            If None it set it to min and max of x, i.e., `xrange = [min(x), max(x)]`

        bins : int, optional
            The numbers of bins to compute the local regression parameters. The default value is 60 bins.

        kernel_type : string, optional
            The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
            If None it uses the pre-specified `kernel_width`

        fast_calc : boolean
            When False, do nothing
            When True , the method only uses data within 3 x kernel_width from the scale mu.
             It speeds up the calculation by removing objects that have extremely small weight.

        Returns
        -------
        numpy array
             Individual residuals.
        """

        if len(X.shape) == 1: X = X[:, None] #Make sure X is atleast 2D

        # Define x_values to compute regression parameters at
        xline = setup_bins(xrange, bins, X[:, 0])
        kernel_width = setup_kernel_width(kernel_width, self.kernel_width, xline.size)

        if kernel_type is None:
            kernel_type = self.kernel_type

        #Get fit
        output = self.fit(X, y, y_err, xrange, bins, nBootstrap, fast_calc, verbose,
                          kernel_type = kernel_type, kernel_width = kernel_width)

        xline, intercept, slopes, scatter = output[0], output[2], output[3], output[4]

        # Edge case, where nBootstrap == 1, or X has only one column
        # Add necessary axes for this edge case to work with code
        if nBootstrap == 1:
            slopes = slopes[np.newaxis, :]

        if X.shape[1] == 1:
            slopes = slopes[:, :, np.newaxis]

        #Select only objects within domain of fit
        Mask = (X[:, 0] >= np.min(xline)) & (X[:, 0] <= np.max(xline))

        Masked_X, Masked_y = X[Mask], y[Mask]

        intercept_interp = interp1d(xline, intercept)(Masked_X[:, 0])
        slopes_interp    = interp1d(xline, np.swapaxes(slopes, 1, 2))(Masked_X[:, 0])
        scatter_interp   = interp1d(xline, scatter)(Masked_X[:, 0])

        mean_y_interp    = intercept_interp + np.sum(slopes_interp * Masked_X.T, axis = 1)

        res = (Masked_y - mean_y_interp)/scatter_interp

        if nBootstrap == 1: res = np.squeeze(res, 0)

        return res

    def outlier_rejection(self, X, Y, sigma, xrange=None, bins=25,
                          fast_calc = False, verbose = False, kernel_type=None, kernel_width=None):
        """
        This simple function uses the normalized residuals, i.e. how many sigma an object is
        from the mean relation < y | X >, to perform outlier rejection. Any object that lives beyond
        a certain sigma range from the mean relation is rejected.

        Parameters
        ----------
        X : numpy array
            Independent variable data vector. Can have multiple features.

        Y : numpy array
            Dependent variable data vector. Can have multiple features but the outlier
            filtering is run on one feature at a time, and the masks are combined at the end.

        xrange : list, optional
            The range of regression. The first element is the min and the second element is the max.
            If None it set it to min and max of x, i.e., `xrange = [min(x), max(x)]`

        bins : int, optional
            The numbers of bins to compute the local regression parameters. The default value is 60 bins.

        kernel_type : string, optional
            The kernel type, ['gaussian', 'tophat'] else it assumes tophat kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'tophat' then 'width' is the width of the tophat kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        1D numpy array
             Mask that is True if object is within sigma range, and false otherwise.
             Same length as X and y. If y has multiple features, then a mask is computed for
             each feature as combined at the end --- an entry in the final mask is True only if
             all features in y lie within the sigma range.
        """

        if len(Y.shape) == 1: Y = Y[:, None]

        Mask = np.ones(len(Y)).astype(bool)

        for i in range(Y.shape[1]):
            res  = self.residuals(X, Y[:, i], None, xrange, bins, 1, fast_calc, verbose, kernel_type, kernel_width)
            Mask = Mask & (np.abs(res) < sigma)

        return Mask
