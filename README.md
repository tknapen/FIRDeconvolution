# FIRDeconvolution
FIRDeconvolution is a python class that performs finite impulse response fitting on time series data, in order to estimate event-related signals. 


Example use cases are fMRI and pupil size analysis. The package performs the linear least squares analysis using numpy.linalg as a backend, but can switch between different backends, such as statsmodels (which is implemented). For very collinear design matrices ridge regression is implemented through the sklearn RidgeCV function. Bootstrap estimates of error regions are implemented through residual reshuffling. 


It is possible to add covariates to the events to estimate not just the impulse response function, but also correlation timecourses with secondary variables. Furthermore, one can add the duration each event should have in the designmatrix, for designs in which the durations of the events vary. 


In neuroscience, the inspection of the event-related signals such as those estimated by FIRDeconvolution is essential for a thorough understanding of one's data. Researchers may overlook essential patterns in their data when blindly running GLM analyses without looking at the impulse response shapes. 


The test notebook explains how the package can be used for data analysis, by creating toy signals and then using FIRDeconvolution to fit the impulse response functions from the toy data. 


## Dependencies
numpy, scipy, matplotlib, statsmodels, sklearn

TODO
- temporal autocorrelation correction









FIRDeconvolution

class FIRDeconvolution(__builtin__.object)
|  Instances of FIRDeconvolutionOperator can be used to perform FIR fitting on time-courses.
|  
|  Methods defined here:
|  
|  __init__(self, signal, events, event_names=[], covariates=None, durations=None, sample_frequency=1.0, deconvolution_interval=[-0.5, 5], deconvolution_frequency=None)
|      FIRDeconvolution takes a signal (signals X nr samples), sampled at sample_frequency in Hz, and deconvolves this signal using least-squares FIR fitting. 
|      The resulting FIR curves are sampled at deconvolution_frequency in Hz, for the interval deconvolution_interval in [start, end] seconds.
|      Event occurrence times are given in seconds.
|      covariates is a dictionary, with keys starting with the event they should be 'attached' to, followed by a _ sign and further name. 
|      The values of the covariate dictionary are numpy arrays with the same length as the original events.
|      The same holds for durations, which are an equal-shape numpy array also. Durations must be small relative to the deconvolution interval length, and given in seconds.
|  
|  add_continuous_regressors_to_design_matrix(self, regressor)
|      add_continuous_regressors_to_design_matrix expects as input a regressor shaped similarly to the design matrix.
|      one uses this addition to the design matrix when one expects the data to contain nuisance factors that aren't tied to the moments of specific events. For instance, in fMRI analysis this allows us to add cardiac / respiratory regressors, as well as tissue and head motion timecourses to the designmatrix.
|      the shape of the regressor argument is required to be (nr_regressors, self.resampled_signal.shape[-1])
|  
|  betas_for_cov(self, covariate='0')
|      betas_for_cov returns the betas associated with a specific covariate.
|      covariate is specified by name.
|  
|  betas_for_events(self)
|      betas_for_events creates an internal self.betas_per_event_type array, of (nr_covariates x self.devonvolution_interval_size), 
|      which holds the outcome betas per event type
|  
|  bootstrap_on_residuals(self, nr_repetitions=1000)
|      bootstrap_on_residuals bootstraps, for nr_repetitions, by shuffling the residuals. 
|      bootstrap_on_residuals should only be used on single-channel data, as otherwise the memory load might 
|      increase too much. This uses the lstsq backend regression for fast fitting across nr_repetitions channels.
|      Please note that shuffling the residuals may change the autocorrelation of the bootstrap samples 
|      relative to that of the original data and that may reduce validity.
|      
|      reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals
|  
|  calculate_rsq(self)
|      calculate_rsq calculates coefficient of determination, or r-squared. 
|      defined here as 1.0 - SS_res / SS_tot.
|      rsq is only calculated for those timepoints in the data for which the design matrix is non-zero
|  
|  create_design_matrix(self)
|      create_design_matrix calls create_event_regressors for each of the covariates in the self.covariates dict. 
|      self.designmatrix is created and is shaped (nr_regressors, self.resampled_signal.shape[-1])
|  
|  create_event_regressors(self, event_times_indices, covariates=None, durations=None)
|      create_event_regressors takes the index of the event for which to create the regressors. 
|      it may or may not be supplied with a set of covariates and durations for these events.
|  
|  predict_from_design_matrix(self, design_matrix)
|      predict_from_design_matrix takes a design matrix (timepoints, betas.shape), 
|      and returns the predicted signal given this design matrix.
|  
|  regress(self, method='lstsq')
|      regress performs linear least squares regression of the designmatrix on the data. 
|      one may choose a method out of the options 'lstsq', 'sm_ols'.
|      this results in the creation of instance variables 'betas' and 'residuals', to be used afterwards.
|  
|  ridge_regress(self, cv=20, alphas=None)
|      perform ridge regression on the design_matrix.
|      for this, we use sklearn's RidgeCV functionality.
|      the cv argument inherits the RidgeCV cv argument's functionality, as does alphas.
|      cv determines the amount of folds of the cross-validation, and alphas is a list of values for alpha, the penalization parameter, to be traversed by the procedure.

