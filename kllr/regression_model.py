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
    |             lm.fit(x, y, nbins=11)                                   |
    |                                                                      |
    ------------------------------------------------------------------------

"""

import numpy as np
from tqdm import tqdm
from sklearn import linear_model

def scatter(X, y, slopes, intercept, y_err = None, dof=None, weight=None):
    """
    This function computes the weighted scatter about the mean relation.
    If weight = None, then this is the regular scatter.

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

    weight : numpy array, optional
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

    #If X is provided as a 1D array then convert to
    #2d array with shape (N, 1)
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

    #Make sure slopes is an 1D array
    slopes = np.atleast_1d(slopes)

    if len(slopes.shape) > 1:
        raise ValueError(
            "Incompatible dimension for slopes. It should be a one dimensional numpy array,"
            ": len(slopes.shape) = %i." %len(slopes.shape))

    if dof is None:
        dof = len(X)

    if y_err is None:
        y_err = 0
    else:
        weight = weight/y_err

    if weight is None:
        sig2 = sum((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2 - y_err**2) / (dof - 1)
    else:
        sig2 = np.average((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2 - y_err**2, weights = weight)
        sig2 /= 1 - np.sum(weight**2)/np.sum(weight)**2 #Required factor for getting unbiased estimate

    if sig2 <= 0:

        print("The uncertainty, y_err, is larger than the instrinsic scatter.",
              "The corrected variance, var_true = var_obs - y_err^2, is negative.")

        return sig2

    else:
        return np.sqrt(sig2)

def moments(m, X, y, slopes, intercept, y_err = None, dof=None, weight=None):
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

    y_err : numpy array, optional
        Uncertainty on dependent variable, y.
        Must contain only non-zero positive values.
        Default is None.

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

    weight : numpy array, optional
        Individual weights for each sample. If None it assumes a uniform weight.


    Returns
    -------
    float, or numpy array
        The weighted moments of the data. A single float if only one moment
        was requested, and a numpy array if multiple were requested.

    """

    #If X is provided as a 1D array then convert to
    #2d array with shape (N, 1)
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

    #Make sure slopes is an 1D array
    slopes = np.atleast_1d(slopes)

    if len(slopes.shape) > 1:
        raise ValueError(
            "Incompatible dimension for slopes. It should be a one dimensional numpy array,"
            ": len(slopes.shape) = %i." %len(slopes.shape))

    if isinstance(y_err, (np.ndarray, list, tuple)):

        y_err = np.asarray(y_err)

        if (y_err <= 0).any():
            raise ValueError("Input y_err contains either zeros or negative values.",
                             "It should contain only positive values.")


    elif y_err is not None:

        weight = weight/y_err

    if dof is None:
        dof = len(X)

    m      = np.atleast_1d(m).astype(int)
    output = np.zeros(m.size)

    residuals = np.array(y) - (np.dot(X, slopes) + intercept)
    for i in range(output.size):

        if weight is None:
            output[i] = np.sum(residuals**m[i]) / dof

        else:
            output[i] = np.average(residuals**m[i], weights=weight)

    if output.size == 1:
        return output[0]

    else:
        return output

def skewness(X, y, slopes, intercept, y_err = None, dof=None, weight=None):
    """
    This function computes the weighted skewness about the mean relation.
    If weight = None, then this is the regular skewness.

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

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The weighted skewness of the sample.
        It is just the standard skewness if Weights = None.

    """

    m2, m3 = moments([2, 3], X, y, slopes, intercept, y_err, dof, weight)

    skew = m3/m2**(3/2)

    return skew

def kurtosis(X, y, slopes, intercept, y_err = None, dof=None, weight=None):
    """
    This function computes the weighted kurtosis about the mean relation.
    If weight = None, then this is the regular skewness.

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

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation

    """

    m2, m4 = moments([2, 4], X, y, slopes, intercept, y_err, dof, weight)

    kurt = m4/m2**2

    return kurt

def calculate_weights(x, kernel_type='gaussian', mu=0, width=0.2):
    """
    According to the provided kernel, this function computes the weight assigned to each data point.

    Parameters
    ----------
    x : numpy array
        A one dimensional data vector.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

    mu, width : float, optional
        If kernel_type = 'gaussian' then 'mu' and 'width' are the mean and width of the gaussian kernels, respectively.
        If kernel_type = 'uniform' then 'mu' and 'width' are the mean and width of the uniform kernels, respectively.

    Returns
    -------
    float
        the weight vector
    """

    if len(x.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X. X  should be one dimensional numpy array,"
            ": len(X.shape) = %i." % (len(x.shape)))

    # the gaussian kernel
    def gaussian_kernel(x, mu=0.0, width=1.0):
        return np.exp(-(x - mu) ** 2 / 2. / width ** 2)

    # the uniform kernel
    def uniform_kernel(x, mu=0.0, width=1.0):
        w = np.zeros(len(x))
        w[(x - mu < width / 2.0) * (x - mu > -width / 2.0)] = 1.0
        return w

    if kernel_type == 'gaussian':
        w = gaussian_kernel(x, mu=mu, width=width)
    elif kernel_type == 'uniform':
        w = uniform_kernel(x, mu=mu, width=width)
    else:
        print("Warning : ", kernel_type, "is not a defined filter.")
        print("It assumes w = 1 for every point.")
        w = np.ones(len(x))

    return w

class kllr_model():
    """
    A class used to represent a KLLR model and perform the fit. It is supported bu additional functions that allows
     to compute the conditional properties such as residuals about the mean relation,
     the correlation coefficient, and the covariance.

    Attributes
    ----------
    kernel_type : string
        The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

    kernel_width : float
         If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
         If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.

    Methods
    -------
    linear_regression(x, y, weight = None)
        perform a linear regression give a set of weights

    subsample(x, length=False)
        generate a bootstrapped sample

    correlation_fixed_x(self, data_x, data_y, data_z, x, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    covariance_fixed_x(x, y, xrange = None, nbins = 60, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    residuals(x, y, xrange = None, nbins = 60, kernel_type = None, kernel_width = None)
        compute residuls about the mean relation i.e., res = y - <y|X>

    PDF_generator(self, res, nbins, nBootstrap = 1000, funcs = {}, xrange = (-4, 4), verbose = True,  **kwargs)
        generate a binned PDF of residuals around the mean relation

    fit(x, y, xrange = None, nbins = 25, kernel_type = None, kernel_width = None)
        fit a kernel localized linear relation to (x, y) pairs, i.e. <y | x> = a(y) x + b(y)

    """

    def __init__(self, kernel_type='gaussian', kernel_width=0.2):
        """
        Parameters
        ----------
        kernel_type : string, optional
            the kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

        kernel_width : float, optional
            if kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            if kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
        """

        self.kernel_type = kernel_type
        self.kernel_width = kernel_width

    def linear_regression(self, X, y, y_err = None, weight=None, compute_skewness = False, compute_kurtosis = False):
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

        weight : float, optional
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

        #if X is not 2D then raise error
        if len(X.shape) > 2:
            raise ValueError("Incompatible dimension for X."
                             "X must be a numpy array with atmost two dimensions.")

        elif len(X.shape) == 1:

            X = X[:, None] #convert 1D to 2D array


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
                raise ValueError("Input y_err contains either zeros or negative values.",
                                 "It should contain only positive values.")

        regr = linear_model.LinearRegression()
        # Train the model using the training sets

        if y_err is None:
            regr.fit(X, y, sample_weight=weight)

        elif (weight is not None) and (y_err is not None):
            regr.fit(X, y, sample_weight=weight/y_err)

        elif (weight is None) and (y_err is not None):
            regr.fit(X, y, sample_weight=1/y_err)

        slopes = regr.coef_
        intercept = regr.intercept_

        sig  = scatter(X, y, slopes, intercept, y_err, weight=weight)

        skew, kurt = None, None #Set some default values

        if compute_skewness: skew = skewness(X, y, slopes, intercept, weight=weight)
        if compute_kurtosis: kurt = kurtosis(X, y, slopes, intercept, weight=weight)

        return intercept, slopes, sig, skew, kurt

    def subsample(self, x, length=False):
        """
        This function re-samples an array and returns the re-sampled array
        and its indices (if you need to use it as a mask for other arrays)

        Parameters
        ----------
        x : numpy array
            One dimensional data array.

        length : bool, optional
            The length of bootstrapped sample. If False it assumes `length = len(x)`.

        Returns
        -------
        numpy array
            the re-sampled vector

        numpy array
            the re-sample index
        """
        x = np.array(x)
        l = length if length else len(x)
        resample = np.floor(np.random.rand(l) * int(len(x))).astype(int)
        return x[resample], resample

    def correlation_fixed_x(self, data_x, data_y, data_z, x, kernel_type=None, kernel_width=None):
        """
        This function computes the conditional correlation between two variables data_y and data_z at point x.

        Parameters
        ----------
        data_x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        data_y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        data_z : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        x : float
            Value of the conditional parameter. It computes the correlation coefficient at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        float
             Correlation coefficient.
        """

        if kernel_type is not None:
            self.kernel_type = kernel_type

        if kernel_width is not None:
            self.kernel_width = kernel_width

        weight = calculate_weights(data_x, kernel_type=self.kernel_type, mu=x, width=self.kernel_width)

        intercept, slope, sig = self.linear_regression(data_x, data_y, weight=weight)[0:3]
        dy = data_y - slope * data_x - intercept

        intercept, slope, sig = self.linear_regression(data_x, data_z, weight=weight)[0:3]
        dz = data_z - slope * data_x - intercept

        sig = np.cov(dy, dz, aweights=weight)

        return sig[1, 0] / np.sqrt(sig[0, 0] * sig[1, 1])

    def covariance_fixed_x(self, data_x, data_y, data_z, x, kernel_type=None, kernel_width=None):
        """
        This function computes the conditional covariance between two variables data_y and data_z at point x.

        Parameters
        ----------
        data_x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        data_y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        data_z : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        x : float
            Value of the conditional parameter. It computes the covariance at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        float
             Covariance.
        """

        if kernel_type is not None:
            self.kernel_type = kernel_type

        if kernel_width is not None:
            self.kernel_width = kernel_width

        weight = calculate_weights(data_x, kernel_type=self.kernel_type, mu=x, width=self.kernel_width)

        intercept, slope, sig = self.linear_regression(data_x, data_y, weight=weight)[0:3]
        dy = data_y - slope * data_x - intercept

        intercept, slope, sig = self.linear_regression(data_x, data_z, weight=weight)[0:3]
        dz = data_z - slope * data_x - intercept

        sig = np.cov(dy, dz, aweights=weight)

        return sig[1, 0]

    def residuals(self, x, y, xrange=None, nbins=60, kernel_type=None, kernel_width=None):
        """
        This function computes the residuals about the mean relation, i.e. res = y - <y | x>.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        xrange : list, optional
            The range of regression. The first element is the min and the second element is the max.
            If None it set it to min and max of x, i.e., `xrange = [min(x), max(x)]`

        nbins : int, optional
            The numbers of bins to compute the local regression parameters. The default value is 60 bins.

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        numpy array
             Individual residuals.
        """

        if kernel_type is not None:
            self.kernel_type = kernel_type

        if kernel_width is not None:
            self.kernel_width = kernel_width

        res = np.array([])  # Array to store residuals
        Index = np.array([])  # array to map what residual belongs to what Halo

        # NOTE: The number of sampling points (60) is currently used as default option
        # changing it only changes accuracy a bit (narrower bins means interpolations are more accurate),
        # and also changes computation time
        if xrange is None:
            xline = np.linspace(np.min(x) - 0.01, np.max(x) + 0.01, nbins)
        else:
            xline = np.linspace(xrange[0], xrange[1], nbins)

        # Loop over each bin defined by the bin edges above
        for i in range(len(xline) - 1):
            # Compute weight at each edge
            w1 = calculate_weights(x, kernel_type=self.kernel_type, mu=xline[i], width=self.kernel_width)
            w2 = calculate_weights(x, kernel_type=self.kernel_type, mu=xline[i + 1], width=self.kernel_width)

            # Compute expected y-value at each bin-edge
            intercept1, slope1, scatter1 = self.linear_regression(x, y, weight=w1)[0:3]
            yline1 = slope1 * xline[i] + intercept1
            intercept2, slope2, scatter2 = self.linear_regression(x, y, weight=w2)[0:3]
            yline2 = slope2 * xline[i + 1] + intercept2

            # Compute slope in this bin
            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])

            # Mask to select only halos in this bin
            mask = (x >= xline[i]) & (x < xline[i + 1])

            # Interpolate to get scatter at each halo
            std = scatter1 + (scatter2 - scatter1) / (xline[i + 1] - xline[i]) * (x[mask] - xline[i])
            # Interpolate expected y-values and Compute residuals
            dy = y[mask] - (yline1 + (yline2 - yline1) / (xline[i + 1] - xline[i]) * (x[mask] - xline[i]))
            res = np.concatenate((res, dy / std))

            # Keep track of an index that maps which residual belongs to which halo
            Index = np.concatenate((Index, np.where(mask)[0]))

        # Reshuffle residuals so that res[i] was computed using the halo with values x[i] and y[i]
        res = np.array(res)[np.argsort(Index)]

        return res

    def PDF_generator(self, res, nbins=20, nBootstrap=1000, funcs={}, xrange=(-4, 4), verbose=True,
                      density=True, weights=None):
        """

        Parameters
        ----------
        res : numpy array
            Individual residuals, i.e. res = y - <y|x>.

        nbins : integer, optional
            Number of bins for the PDF.

        xrange : list, optional
            Tuple containing min and max bin values.

        nBootstrap : integer, optional
            Number of Bootstrap realizations of the PDF.

        funcs : dictionary, optional
            Dictionary of functions to apply on the Bootstrapped residuals. Format is {'Name': func}.

        verbose : bool, optional
            Turn on/off the verbosity of the PDF output during the bootstrapping.

        density : bool, optional
            If False, the result will contain the number of samples in each bin.
             If True, the result is the value of the probability density function at the bin, normalized such
             that the integral over the range is 1. Note that the sum of the histogram values will not be
             equal to 1 unless bins of unity width are chosen; it is not a probability mass function.

        weights : numpy array, optional
            An array of weights, of the same shape as a. Each value in a only contributes its associated weight
             towards the bin count (instead of 1). If density is True, the weights are normalized, so that the
             integral of the density over the range remains 1. If None it assumes a uniform weight.

        Returns
        -------
        numpy array
            Numpy array of size (nBootstrap, nbins) containing all realizations of PDFs

        numpy array
            Central values of the bins of the PDF

        Dictionary
            Dictionary with format {'name': result}, where result is the output of user-inputted inputted
            functions acting on residuals.
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

        for iBoot in iterations:

            if iBoot == 0:
                residuals = res
                try:
                    w = weights
                except:
                    w = None
            else:
                residuals, index = self.subsample(res)
                try:
                    # If weights exist, reshuffle them according to subsampling
                    w = weights[index]
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

    def fit(self, X, y, y_err = None, xrange=None, nbins=25, compute_skewness = False, compute_kurtosis = False,
            kernel_type=None, kernel_width=None):
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

        nbins : int, optional
            The numbers of data points to compute the local regression parameters

        compute_skewness : boolean, optional
            If compute_skewness == True, the weighted skewness
            is computed and returned in the output

        compute_kurtosis : boolean, optional
            If compute_kurtosis == True, the weighted kurtosis
            is computed and returned in the output

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
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
        if xrange is None:

            xline = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), nbins)
        else:
            xline = np.linspace(xrange[0], xrange[1], nbins)

        if kernel_width is not None:
            self.kernel_width = kernel_width

        if kernel_type is not None:
            self.kernel_type = kernel_type

        # Generate array to store output from fit
        yline, slope, intercept, scatter, skew, kurt = [np.zeros(xline.size) for i in range(6)]

        # loop over every sample point
        for i in range(len(xline)):

            # Generate weights at that sample point
            w = calculate_weights(X[:, 0], kernel_type=self.kernel_type, mu=xline[i], width=self.kernel_width)

            # Compute fit params using linear regressions
            output = self.linear_regression(X, y, y_err, w, compute_skewness, compute_kurtosis)
            intercept[i] = output[0]
            slope[i]     = output[1][0]
            scatter[i]   = output[2]
            skew[i]      = output[3]
            kurt[i]      = output[4]

            # Generate expected y_value using fit params
            yline[i] = slope[i] * xline[i] + intercept[i]

        return xline, yline, intercept, slope, scatter, skew, kurt

    def multivariate_fit(self, X, y, y_err = None, xrange=None, nbins=25,
                         compute_skewness = False, compute_kurtosis = False,
                         kernel_type=None, kernel_width=None):
        """
        This function computes the local regression parameters at the points within xrange.
        This version support supports multidimensional data for the independent data matrix, X.
        However the kernel weighting uses only the first column in X, i.e. X[:, 0].
        The predicted variable, y, must still be 1D.

        Parameters
        ----------
        X : numpy array (n_rows, n_features)
            Independent variable data matrix.
            The weighting is only applied to the first feature.

        y : numpy array (n_rows)
            Dependent variable data vector. Must be a one dimensional data vector.

        y_err : numpy array (n_rows), optional
            Uncertainty on dependent variable, y.
            Must contain only non-zero positive values.
            Default is None.

        xrange : list, optional
            The first element is the min and the second element is the max,
            If None, it sets xrange to [min(x), max(x)]

        nbins : int, optional
            The numbers of data points to compute the local regression parameters

        compute_skewness : boolean, optional
            If compute_skewness == True, the weighted skewness
            is computed and returned in the output

        compute_kurtosis : boolean, optional
            If compute_kurtosis == True, the weighted kurtosis
            is computed and returned in the output

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`.

        Returns
        -------

        numpy-array
            The slope w.r.t each input feature,
            evaluated at the local points of the first feature

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

        # Define x_values to compute regression parameters at
        # Only use first feature in independent variable matrix
        if xrange is None:
            xrange = (np.min(X[:, 0]), np.max(X[:, 0]))
        elif xrange[0] is None:
            xrange[0] = np.min(X[:, 0])
        elif xrange[1] is None:
            xrange[1] = np.max(X[:, 0])

        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

        if kernel_width is not None:
            self.kernel_width = kernel_width

        if kernel_type is not None:
            self.kernel_type = kernel_type

        # Generate array to store output from fit
        slopes = np.zeros(shape=(xline.size, X.shape[1]))
        intercept, scatter, skew, kurt = [np.zeros(xline.size) for i in range(4)]

        # loop over every sample point
        for i in range(xline.size):

            # Generate weights at that sample point
            w = calculate_weights(X[:, 0], kernel_type=self.kernel_type, mu=xline[i], width=self.kernel_width)

            # Compute fit params using multivariate linear regression
            output = self.linear_regression(X, y, y_err, w, compute_skewness, compute_kurtosis)

            intercept[i] = output[0]
            slopes[i]    = output[1]
            scatter[i]   = output[2]
            skew[i]      = output[3]
            kurt[i]      = output[4]

            # Doesn't seem clear on what it means to
            # compute mean relation just as a function
            # of M200c even though we regress on all properties
            # so I don't give expected <y | x_0> here (eg. x_0 --> M200c)

        return xline, slopes, scatter, skew, kurt
