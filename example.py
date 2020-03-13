from kllr import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

print(sys.path)

df = pd.read_csv('./data/TNG300_Halos.csv')
print(df.columns, np.min(df.M200))
x = np.array(df.M200)
y = np.array(df.MGas)


# lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)
# xrange, yrange_mean, intercept, slope, scatter = lm.fit(x, y, xrange=[13.5, 14.5], nbins=11)
# print(slope)

# data, ax = Plot_Fit(df, 'M200', 'MGas', show_data=True, ax=None) ## PASSED
# data, ax = Plot_Fit(df, 'M200', 'MStar', show_data=True, cutoff=13.5, ax=ax) ## PASSED
data, ax = Plot_Fit_Split(df, 'M200', 'MStar_BCG100', 'z_form', split_bins=3) ## PASSED
# data, ax = Plot_Fit_Params(df, 'M200', 'MGas') ## PASSED
# data, ax = Plot_Fit_Params_Split(df, 'M200', 'MGas', 'z_form', split_bins = [0.0, 0.2, 0.3, 0.6])
# data, ax = Plot_Correlation(df, 'M200', 'MGas', 'MStar') ## PASSED
# data, ax = Plot_Correlation_Split(df, 'M200', 'MGas', 'MStar', 'z_form', split_bins = [0.0, 0.2, 0.3, 0.6]) ##PASSED
# data, ax = Plot_Covariance(df, 'M200', 'MGas', 'MStar', GaussianWidth=0.2)
# data, ax = Plot_Covariance_Split(df, 'M200', 'MGas', 'MStar', 'z_form', split_bins = [0.0, 0.2, 0.3, 0.6]) ## PASSED
# ax = Plot_Correlation_Matrix(df, 'M200', ['MGas', 'MGas_T', 'sigma_DM_3D']) ## PASSED
# ax = Plot_Correlation_Matrix_Split(df, 'M200', ['MGas', 'MGas_T', 'sigma_DM_3D'], 'z_form', split_bins=[0.0, 0.2, 0.3, 0.6]) ## PASSED
# ax = Plot_Covariance_Matrix(df, 'M200', ['MGas', 'MStar', 'sigma_DM_3D']) ## PASSED
# ax = Plot_Covariance_Matrix_Split(df, 'M200', ['MGas', 'MStar', 'MStar_BCG100'], 'z_form', split_bins=[0.0, 0.2, 0.3, 0.6]) ## PASSED
# data, ax = Plot_Residual(df, 'M200', 'MGas') ## PASSED
# data, ax = Plot_Residual_Split(df, 'M200', 'MGas', 'z_form', split_bins = [0.0, 0.2, 0.3, 0.6])

plt.show()