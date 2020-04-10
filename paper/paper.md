---
title: 'KLLR: A Kernel Localized Linear Regression method'
tags:
  - Python
  - Linear Regression
  - Covariance Analysis
authors:
  - name: Arya Farahi
    orcid: 0000-0003-0777-4618
    affiliation: 1
  - name: Dhayaa Anbajagane
    affiliation: 2
    orcid: 0000-0003-3312-909X
  - name: August E. Evrard
    orcid: 0000-0002-4876-956X
    affiliation: 2
affiliations:
 - name: Michigan Institute for Data Science, University of Michigan, Ann Arbor, MI 48109, USA
   index: 1
 - name: Department of Physics and Leinweber Center for Theoretical Physics, University of Michigan, Ann Arbor, MI 48109, USA
   index: 2
date: 15 April 2020
bibliography: paper.bib

---

# Summary

Linear regression of the simple least-squares variety has been a canonical method used
to characterize the relation between two variables, one dependent and one independent.
But its utility is limited by the fact that it reduces full population statistics down
to three numbers: a slope, normalization and variance/standard deviation. With large
observational or simulated samples we can perform a more sensitive analysis using a localized
linear regression method (e.g., [@Farahi:2018; @Anbajagane:2020]). Thus, our objective is to find
a locally linear, but globally non-linear, relation between a pair of random variables $X$
and $Y$ where $Y$ can be a multi-dimentional data. A popular approach is Gaussian Processes
[@Alvarez:2011]. Another popular approach is [@cleveland1979robust] local regression with
a weighted least-square fitting where the weights are assigned with a kernel or weight function.
The Kernel Localized Linear Regression (KLLR) is based on a variation of local linear regression
[@cleveland1979robust]. KLLR method generates estimates of conditional statistics in terms of
the local slope, normalization, and covariance. Such a method provides a more nuanced
description of population statistics appropriate for the very large samples with non-linear
trends.

`KLLR` is a Python package for multivariate regression analysis, and is an implementation
of the kernel weighted linear regression that performs a localized Linear regression
described in @Farahi:2018. The main function, `kllr_model(...)`, reports the local normalization,
slope, and standard deviation at each point in $X$. The current implementation of `kllr_model`
supports a uniform and a Gaussian kernel with width defined by the user. It employs the bootstrap
re-sampling technique to estimate the uncertainties for each model parameter. `KLLR` is backed
by a set of user-friendly and fast visualization tools so practitioners can seamlessly generate
informative data summaries and visualizations. The `KLLR` package is indexed by Python Package
Index (PyPI) and can be installed through `pip install`.

This package is designed and implemented with applications in data analysis of astronomical
datasets in mind, but its applications is not limited to astronomy domain.  


# Use cases

`KLLR` enables the practitioners to perform multivariate regression analysis and generate
informative visualizations. The visualization modules seamlessly fit the `KLLR` model to a set
of data, estimate the uncertainties, and produce a set of summary statistics visualizations.

When the dependent variable, $Y$, is one dimensional, the user can plot the best fit, the local
slope, and standard deviation as a function of the independent variable, $X$, and also the normalized
residuals in $Y$. The user is encouraged to investigate the normalized residuals, since the model assumes the
conditional statistics of the data follow a multi-variate normal distribution. If there is any evidence
of strong skewness or fatter or narrower tail than expected from the normal distribution, the `KLLR`
model might be inappropriate. When the dependent variable, $Y$, is multi-dimensional, the user can
generate---on top of the previously-mentioned one-dimensional features---a visualization of the conditional covariance
and correlation matrix.

Another important feature of `KLLR` is that the user can split their dataset into non-overlapping
subsets based on a third quantity, $Z$. And then perform multivariate regression analysis for each
subset independently and visualize the summary statistics on a same plot. This feature has been
used in XX. With the code, we provide a set of examples to illustrate different use cases of `KLLR`.


# External libraries

The KLLR method is implemented through `NumPy` [@NumPy] and `Scikit-learn` library [@Scikitlearn],
and plotting modules use `Pandas` [@Pandas] data structure to perform KLLR
and visualize through `Matplotlib` [@Matplotlib].


# Acknowledgements

Arya Farahi is supported by a Michigan Institute for Data Science Fellowship.
