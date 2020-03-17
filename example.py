'''
This module generates/simulates a fake dataset of (4 + 1) properties,
where they are all correlated with each other, and runs the LLR
modules and the plotting modules on the fake data.
'''

from kllr import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


N        = 10000 # Number of samples to be generated
width    = 0.4   # Kernel_width used in all the plotting functions
saveplot = False #If True, save plots to ./examples/

# Generate 2D vector to hold all properties
# This will help with matrix multiplication later
S = np.zeros((5, N))

# Generate first variable
# We will assume scaling relations y = pi + alpha*x
# to generate the other variables
x = np.random.uniform(-10, 10, N)

# Specify intercept and slopes of other variables
pi1, alpha1, alpha11 = 0, 1, 1
pi2, alpha2 = 0, 1
pi3, alpha3, alpha31 = 0, 1, -1
pi4, alpha4 = 0, 1

# Generate other variables
y1 = pi1 + (alpha1+alpha11*x)*x
y2 = pi2 + alpha2*x
y3 = pi3 + (alpha3+alpha31*x)*x
y4 = pi4 + alpha4*x

# Assign variables to 2D array
S[0, :] = x
S[1, :] = y1
S[2, :] = y2
S[3, :] = y3
S[4, :] = y4

# Covariance matrix
# TODO: x-dependent covariance matrix!
Cov = np.array([[16., 0.,  4., 0.],
                [0.,  4., -1., 0.],
                [4., -1.,  4., 0.],
                [0.,  0.,  0., 1.]])

# Generate N-D gaussian variable
# Assuming scatter around mean relations is Gaussian
delta = np.random.multivariate_normal(np.zeros(4), Cov, N).T

# Account for covariance in S matrix
S[1:] += delta

df = pd.DataFrame(S.T, columns = ['x', 'y1', 'y2', 'y3', 'y4'])

# Setup kernel local linear regression model
lm = kllr_model(kernel_type = 'gaussian', kernel_width = width)

# Compute regression parameters
x, y_exp, intercept_exp, slope_exp, scatter_exp = lm.fit(df['x'], df['y1'], xrange=[2, 8], nbins=11)

# TODO: Add checkdir commands to make sure folder exists, if not make one
# Generate and save fiducial analyses plots
data, ax = Plot_Fit(df, 'x', 'y1', show_data=True, kernel_width = width)
data, ax = Plot_Fit(df, 'x', 'y2', show_data=True, kernel_width = width, ax = ax)
plt.grid()
if saveplot: plt.savefig("./examples/Fit.pdf", bbox_inches='tight')

data, ax = Plot_Fit_Split(df, 'x', 'y1', 'y3', split_mode = 'Residuals', split_bins=3, kernel_width = width)
if saveplot: plt.savefig("./examples/Fit_split.pdf", bbox_inches='tight')

data, ax = Plot_Fit_Params(df, 'x', 'y1', xlog=False, kernel_width = width)
data, ax = Plot_Fit_Params(df, 'x', 'y2', xlog=False, kernel_width = width, ax=ax)
data, ax = Plot_Fit_Params(df, 'x', 'y3', xlog=False, kernel_width = width, ax=ax)
if saveplot: plt.savefig("./examples/Fit_Params.pdf", bbox_inches='tight')

data, ax = Plot_Fit_Params_Split(df, 'x', 'y1', 'y3', split_bins = 2, split_mode = 'Residuals', kernel_width = width)
if saveplot: plt.savefig("./examples/Fit_Params_Split.pdf", bbox_inches='tight')

ax = Plot_Cov_Corr_Matrix(df, 'x', ['y1', 'y2', 'y3'], Output_mode = 'corr', kernel_width = width)
if saveplot: plt.savefig("./examples/Corr_Matrix.pdf", bbox_inches='tight')

ax = Plot_Cov_Corr_Matrix_Split(df, 'x', ['y1', 'y2', 'y3'], 'y3', split_bins = 2,
                                split_mode = 'Residuals', Output_mode = 'corr', kernel_width = width)
if saveplot: plt.savefig("./examples/Corr_Matrix_Split.pdf", bbox_inches='tight')

ax = Plot_Cov_Corr_Matrix(df, 'x', ['y1', 'y2', 'y3'], Output_mode = 'cov', kernel_width = width)
if saveplot: plt.savefig("./examples/Cov_Matrix.pdf", bbox_inches='tight')

ax = Plot_Cov_Corr_Matrix_Split(df, 'x', ['y1', 'y2', 'y3'], 'y3', split_bins = 2,
                                split_mode = 'Residuals', Output_mode = 'cov', kernel_width = width)
if saveplot: plt.savefig("./examples/Cov_Matrix_Split.pdf", bbox_inches='tight')

data, ax = Plot_Residual(df, 'x', 'y1')
if saveplot: plt.savefig("./examples/PDF.pdf", bbox_inches='tight')

data, ax = Plot_Residual_Split(df, 'x', 'y2', 'y3', split_bins = 2, split_mode = 'Residuals', kernel_width = width)
if saveplot: plt.savefig("./examples/PDF_Split.pdf", bbox_inches='tight')

plt.show()
