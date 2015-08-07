# FIRDeconvolution
Python class that performs finite impulse response fitting on time series data, in order to estimate event-related activations. 
It is possible to add covariates to the events, to estimate not just the impulse response function, but also correlation timecourses with other variables.

Example use cases are fMRI and pupil analysis. The package performs the linear least squares analysis using numpy.linalg as a backend. 
