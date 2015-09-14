# FIRDeconvolution
Python class that performs finite impulse response fitting on time series data, in order to estimate event-related activations. 
It is possible to add covariates to the events to estimate not just the impulse response function, but also correlation timecourses with other variables. 
Furthermore, one can add the duration each event should have in the designmatrix. Thus, if events have varying durations, this allows one to estimate the IRF regardless (assuming linearity of course).

Example use cases are fMRI and pupil size analysis. The package performs the linear least squares analysis using numpy.linalg as a backend, but can switch between different backends, such as statsmodels (which is implemented). For very collinear designmatrices, i.e. when a lot of covariates are taken into account, ridge regression is implemented through the sklearn RidgeCV function.

The test notebook in this folder explains how the package can be used for data analysis, by creating signals and then using FIRDeconvolution to fit the impulse response functions that have been put in. 

## Dependencies
numpy, scipy, matplotlib, statsmodels for fitting functionality extension

TODO
- temporal autocorrelation correction, perhaps?
