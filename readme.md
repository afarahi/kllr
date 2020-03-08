<p align="center">
  <img src="logo.png" width="300" title="logo">
</p>

# Introduction

Kernel Localized Linear Regression (KLLR). The module perform the Kernel Localized Linear Regression 
to estimate the local slope and scatter of a non-linear relation.
It perform bootstrap algorithm to estimate the uncertainties, and generate a visualization 
 of the model parameters.

## Dependencies

`numpy`, `matplotlib`, `pandas`, `pathlib`

## Cautions

- ....

## References

[1]. Farahi, Arya, et al. "Localized massive halo properties in BAHAMAS and MACSIS simulations: scalings, lognormality, and covariance." Monthly Notices of the Royal Astronomical Society 478.2 (2018): 2618-2632.

[2]. Anbajagane, Dhayaa, et al. Stellar Property Statistics of Massive Halos from Cosmological Hydrodynamics Simulations: Common Kernel Shapes. No. arXiv: 2001.02283. 2020.


## Acknowledgment


## Quickstart

To start using KLLR, simply use `from kllr import regression` to
access the primary function. The exact requirements for the inputs are
listed in the docstring of the regression() function further below.
An example for using TATTER looks like this:

      from kllr import kllr_model
       
       lm = kllr_model()
       xrange, yrange_mean, intercept, slope, scatter =
           lm.fit(x, y, [0.0, 1.0], nbins=11, GaussianWidth=0.2)


