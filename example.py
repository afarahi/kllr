'''
This module generates/simulates a fake dataset of (4 + 1) properties,
where they are all correlated with each other, and runs the LLR
modules and the plotting modules on the fake data.
'''

from kllr.kllr import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Number of samples to be generated
N = 10000

#Generate 2D vector to hold all properties
#This will help with matrix multiplication later
S = np.zeros((5, N))

#Generate first variable
#We will assume scaling relations y = pi + alpha*x
#to generate the other variables
x = np.random.rand(N)*10

#Specify intercept and slopes of other variables
#TODO: Alpha must be x-dependent!
pi1, alpha1 = 5, 1
pi2, alpha2 = 5, 1
pi3, alpha3 = 5, 1
pi4, alpha4 = 5, 1

#Generate other variables
y1 = pi1 + alpha1*x
y2 = pi2 + alpha2*x
y3 = pi3 + alpha3*x
y4 = pi4 + alpha4*x

#Assign variables to 2D array
S[0, :] = x
S[1, :] = y1
S[2, :] = y2
S[3, :] = y3
S[4, :] = y4

#Covariance matrix
#TODO: x-dependent covariance matrix!
Cov = np.array([[1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 2.,-1., 0.],
                [0., 0.,-1., 1., 0.],
                [0., 0., 0., 0., 1.]])

#Generate N-D gaussian variable
#Assuming scatter around mean relations is Gaussian
delta = np.random.multivariate_normal(np.zeros(5), Cov, N).T

#Account for covariance in S matrix
S += delta

df = pd.DataFrame(S.T, columns = ['Data1', 'Data2', 'Data3', 'Data4', 'Data5'])

#Setup kernel local linear regression model
lm = kllr_model(kernel_type = 'gaussian', kernel_width = 1)

#Compute parameters
x, y_exp, intercept_exp, slope_exp, scatter_exp = lm.fit(df['Data1'], df['Data2'], xrange=[2, 8], nbins=11)

#Add checkdir commands to make sure folder exists

Params = np.concatenate((x, y_exp, intercept_exp, slope_exp, scatter_exp))

np.savetxt("Params.txt", Params)

#Generate and save fiducial analyses plots
data, ax = Plot_Fit(df, 'Data1', 'Data2', show_data=True, ax=None, kernel_width = 1)
plt.savefig("Fit.pdf")

data, ax = Plot_Fit_Split(df, 'Data1', 'Data2', 'Data3', split_bins=[1, 10, 20], kernel_width = 1)
plt.savefig("Fit_split.pdf")

data, ax = Plot_Fit_Params(df, 'Data1', 'Data2', xlog=False, kernel_width = 1)
plt.savefig("Fit_Params.pdf")

data, ax = Plot_Fit_Params_Split(df, 'Data1', 'Data2', 'Data3', split_bins=[1, 10, 20], split_mode='Data', kernel_width = 1)
plt.savefig("Fit_Params_Split.pdf")

ax = Plot_Cov_Corr_Matrix(df, 'Data1', ['Data2', 'Data3', 'Data4'], Output_mode = 'corr', kernel_width = 1)
plt.savefig("Corr_Matrix.pdf")

ax = Plot_Cov_Corr_Matrix_Split(df, 'Data1', ['Data2', 'Data4', 'Data5'], 'Data3', split_bins=[1, 10, 20], Output_mode = 'corr', kernel_width = 1)
plt.savefig("Corr_Matrix_Split.pdf")

data, ax = Plot_Residual(df, 'Data1', 'Data2')
plt.savefig("PDF.pdf")

data, ax = Plot_Residual_Split(df, 'Data1', 'Data2', 'Data3', split_bins=[1, 10, 20], split_mode = 'Data', kernel_width = 1)
plt.savefig("PDF_Split.pdf")

plt.show()
