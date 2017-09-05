#!/usr/bin/env python
# encoding: utf-8
"""
FIRDeconvolution is a python class that performs finite impulse response fitting on time series data, 
in order to estimate event-related signals. These signals can come from any source, but the most likely
source in our experience is some sort of physiological signal such as fMRI voxels, 
galvanic skin response, (GSR) or pupil size recordings. 

The repo for FIRDeconvolution is at https://github.com/tknapen/FIRDeconvolution, 
and the GitHub FIRDeconvolution website is located at http://tknapen.github.io/FIRDeconvolution/.

"""
from __future__ import division

import unittest
import logging

import math
import numpy as np
import scipy as sp
import scipy.signal

import numpy.linalg as LA
from sklearn import linear_model

from IPython import embed as shell


class FIRDeconvolution(object): 
    """Instances of FIRDeconvolution can be used to perform FIR fitting on time-courses. 
    Since many of the computation's parameters are set in the constructor, 
    it is likely easiest to create new instances for each separate analysis you run.
    """

    def __init__(self, signal, events, event_names = [], covariates = None, durations = None, sample_frequency = 1.0, deconvolution_interval = [-0.5, 5], deconvolution_frequency = None):
        """FIRDeconvolution takes a signal and events in order to perform FIR fitting of the event-related responses in the signal. 
        Most settings for the analysis are set here. 

            :param signal: input signal. 
            :type signal: numpy array, (nr_signals x nr_samples)
            :param events: event occurrence times. 
            :type events: list of numpy arrays, (nr_event_types x nr_events_per_type)
            :param event_names: event names. 
            :type events: list of strings, if empty, event names will be string representations of range(nr_event_types)
            :param covariates: covariates belonging to event_types. If None, covariates with a value of 1 for all events are created and used internally.
            :type covariates: dictionary, with keys "event_type.covariate_name" and values numpy arrays, (nr_events)
            :param durations: durations belonging to event_types. If None, durations with a value of 1 sample for all events are created and used internally.
            :type durations: dictionary, with keys "event_type" and values numpy arrays, (nr_events)
            :param sample_frequency: input signal sampling frequency in Hz, standard value: 1.0
            :type sample_frequency: float
            :param deconvolution_interval: interval of time around the events for which FIR fitting is performed.
            :type deconvolution_interval: list: [float, float]
            :param deconvolution_frequency: effective frequency in Hz at which analysis is performed. If None, identical to the sample_frequency.
            :type deconvolution_frequency: float
        
            :returns: Nothing, but the created FIRDeconvolution object.
        """

        self.logger = logging.getLogger('FIRDeconvolution')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.debug('initializing deconvolution with signal sample freq %2.2f, etc etc.' % (sample_frequency))

        self.signal = signal 
        if len(self.signal.shape) == 1:
            self.signal = self.signal[np.newaxis, :]

        # construct names for each of the event types
        if event_names == []:
            self.event_names = [str(i) for i in np.arange(len(events))] 
        else:
            self.event_names = event_names
        assert len(self.event_names) == len(events), \
                    'number of event names (%i, %s) does not align with number of event definitions (%i)' %(len(self.event_names), self.event_names, len(events))
        # internalize event timepoints aligned with names
        self.events = dict(zip(self.event_names, events))

        self.sample_frequency = sample_frequency
        self.deconvolution_interval = deconvolution_interval
        if deconvolution_frequency is None:
            self.deconvolution_frequency = sample_frequency
        else:
            self.deconvolution_frequency = deconvolution_frequency

        self.resampling_factor = self.sample_frequency/self.deconvolution_frequency     
        self.deconvolution_interval_size = np.round((self.deconvolution_interval[1] - self.deconvolution_interval[0]) * self.deconvolution_frequency)
        if not np.allclose([round(self.deconvolution_interval_size)], [self.deconvolution_interval_size]):
            print('self.deconvolution_interval_size, %3.6f should be integer. I don\'t know why, but it\'s neater.'%self.deconvolution_interval_size)
        self.deconvolution_interval_size = int(self.deconvolution_interval_size)
        self.deconvolution_interval_timepoints = np.linspace(self.deconvolution_interval[0],self.deconvolution_interval[1],self.deconvolution_interval_size)

        # duration of signal in seconds and at deconvolution frequency
        self.signal_duration = self.signal.shape[-1] / self.sample_frequency
        self.resampled_signal_size = int(self.signal_duration*self.deconvolution_frequency)
        self.resampled_signal = scipy.signal.resample(self.signal, self.resampled_signal_size, axis = -1)

        # if no covariates, we make a new covariate dictionary specifying only ones.
        # we will loop over these covariates instead of the event list themselves to create design matrices
        if covariates == None:
            self.covariates = dict(zip(self.event_names, [np.ones(len(ev)) for ev in events]))
        else:
            self.covariates = covariates

        if durations == None:
            self.durations = dict(zip(self.event_names, [np.ones(len(ev))/deconvolution_frequency for ev in events]))
        else:
            self.durations = durations

        self.number_of_event_types = len(self.covariates)
        # indices of events in the resampled signal, keeping this as a list instead of an array     
        # at this point we take into account the offset encoded in self.deconvolution_interval[0]       
        
        self.event_times_indices = dict(zip(self.event_names, [((ev + self.deconvolution_interval[0])*self.deconvolution_frequency).astype(int) for ev in events]))
        # convert the durations to samples/ indices also
        self.duration_indices = dict(zip(self.event_names, [(self.durations[ev]*self.deconvolution_frequency).astype(int) for ev in self.event_names]))

    def create_event_regressors(self, event_times_indices, covariates = None, durations = None):
        """create_event_regressors creates the part of the design matrix corresponding to one event type. 

            :param event_times_indices: indices in the resampled data, on which the events occurred.
            :type event_times_indices: numpy array, (nr_events)
            :param covariates: covariates belonging to this event type. If None, covariates with a value of 1 for all events are created and used internally.
            :type covariates: numpy array, (nr_events)
            :param durations: durations belonging to this event type. If None, durations with a value of 1 sample for all events are created and used internally.
            :type durations: numpy array, (nr_events)
            :returns: This event type's part of the design matrix.
        """

        # check covariates
        if covariates is None:
            covariates = np.ones(self.event_times_indices.shape)

        # check/create durations, convert from seconds to samples time, and compute mean duration for this event type.
        if durations is None:
            durations = np.ones(self.event_times_indices.shape)
        else:
            durations = np.round(durations*self.deconvolution_frequency).astype(int)
        mean_duration = np.mean(durations)

        # set up output array
        regressors_for_event = np.zeros((self.deconvolution_interval_size, self.resampled_signal_size))

        # fill up output array by looping over events.
        for cov, eti, dur in zip(covariates, event_times_indices, durations):
            valid = True
            if eti < 0:
                self.logger.debug('deconv samples are starting before the data starts.')
                valid = False
            if eti+self.deconvolution_interval_size > self.resampled_signal_size:
                self.logger.debug('deconv samples are continuing after the data stops.')
                valid = False
            if eti > self.resampled_signal_size:
                self.logger.debug('event falls outside of the scope of the data.')
                valid = False

            if valid: # only incorporate sensible events.
                # calculate the design matrix that belongs to this event.
                this_event_design_matrix = (np.diag(np.ones(self.deconvolution_interval_size)) * cov)
                over_durations_dm = np.copy(this_event_design_matrix)
                if dur > 1: # if this event has a non-unity duration, duplicate the stick regressors in the time direction
                    for d in np.arange(1,dur):
                        over_durations_dm[d:] += this_event_design_matrix[:-d]
                    # and correct for differences in durations between different regressor types.
                    over_durations_dm /= mean_duration
                # add the designmatrix for this event to the full design matrix for this type of event.
                regressors_for_event[:,eti:int(eti+self.deconvolution_interval_size)] += over_durations_dm
        
        return regressors_for_event

    def create_design_matrix(self, demean = False, intercept = True):
        """create_design_matrix calls create_event_regressors for each of the covariates in the self.covariates dict. self.designmatrix is created and is shaped (nr_regressors, self.resampled_signal.shape[-1])
        """
        self.design_matrix = np.zeros((int(self.number_of_event_types*self.deconvolution_interval_size), self.resampled_signal_size))

        for i, covariate in enumerate(self.covariates.keys()):
            # document the creation of the designmatrix step by step
            self.logger.debug('creating regressor for ' + covariate)
            indices = np.arange(i*self.deconvolution_interval_size,(i+1)*self.deconvolution_interval_size, dtype = int)
            # here, we implement the dot-separated encoding of events and covariates
            if len(covariate.split('.')) > 0:
                which_event_time_indices = covariate.split('.')[0]
            else:
                which_event_time_indices = covariate
            self.design_matrix[indices] = self.create_event_regressors( self.event_times_indices[which_event_time_indices], 
                                                                        self.covariates[covariate], 
                                                                        self.durations[which_event_time_indices])

        if demean:
            # we expect the data to be demeaned. 
            # it's an option whether the regressors should be, too
            self.design_matrix = (self.design_matrix.T - self.design_matrix.mean(axis = -1)).T
        if intercept:
            # similarly, intercept is a choice.
            self.design_matrix = np.vstack((self.design_matrix, np.ones((1,self.design_matrix.shape[-1]))))
        
        self.logger.debug('created %s design_matrix' % (str(self.design_matrix.shape)))

    def add_continuous_regressors_to_design_matrix(self, regressors):
        """add_continuous_regressors_to_design_matrix appends continuously sampled regressors to the existing design matrix. One uses this addition to the design matrix when one expects the data to contain nuisance factors that aren't tied to the moments of specific events. For instance, in fMRI analysis this allows us to add cardiac / respiratory regressors, as well as tissue and head motion timecourses to the designmatrix.
        
            :param regressors: the signal to be appended to the design matrix.
            :type regressors: numpy array, with shape equal to (nr_regressors, self.resampled_signal.shape[-1])
        """
        previous_design_matrix_shape = self.design_matrix.shape
        if len(regressors.shape) == 1:
            regressors = regressors[np.newaxis, :]
        if regressors.shape[1] != self.resampled_signal.shape[1]:
            self.logger.warning('additional regressor shape %s does not conform to designmatrix shape %s' % (regressors.shape, self.resampled_signal.shape))
        # and, an vstack append
        self.design_matrix = np.vstack((self.design_matrix, regressors))
        self.logger.debug('added %s continuous regressors to %s design_matrix, shape now %s' % (str(regressors.shape), str(previous_design_matrix_shape), str(self.design_matrix.shape)))

    def regress(self, method = 'lstsq'):
        """regress performs linear least squares regression of the designmatrix on the data. 

            :param method: method, or backend to be used for the regression analysis.
            :type method: string, one of ['lstsq', 'sm_ols']
            :returns: instance variables 'betas' (nr_betas x nr_signals) and 'residuals' (nr_signals x nr_samples) are created.
        """

        if method is 'lstsq':
            self.betas, residuals_sum, rank, s = LA.lstsq(self.design_matrix.T, self.resampled_signal.T)
            self.residuals = self.resampled_signal - self.predict_from_design_matrix(self.design_matrix)
        elif method is 'sm_ols':
            import statsmodels.api as sm

            assert self.resampled_signal.shape[0] == 1, \
                    'signal input into statsmodels OLS cannot contain multiple signals at once, present shape %s' % str(self.resampled_signal.shape)
            model = sm.OLS(np.squeeze(self.resampled_signal),self.design_matrix.T)
            results = model.fit()
            # make betas and residuals that are compatible with the LA.lstsq type.
            self.betas = np.array(results.params).reshape((self.design_matrix.shape[0], self.resampled_signal.shape[0]))
            self.residuals = np.array(results.resid).reshape(self.resampled_signal.shape)

        self.logger.debug('performed %s regression on %s design_matrix and %s signal' % (method, str(self.design_matrix.shape), str(self.resampled_signal.shape)))

    def ridge_regress(self, cv = 20, alphas = None ):
        """perform k-folds cross-validated ridge regression on the design_matrix. To be used when the design matrix contains very collinear regressors. For cross-validation and ridge fitting, we use sklearn's RidgeCV functionality. Note: intercept is not fit, and data are not prenormalized. 

            :param cv: cross-validated folds, inherits RidgeCV cv argument's functionality.
            :type cv: int, standard = 20
            :param alphas: values of penalization parameter to be traversed by the procedure, inherits RidgeCV cv argument's functionality. Standard value, when parameter is None, is np.logspace(7, 0, 20)
            :type alphas: numpy array, from >0 to 1. 
            :returns: instance variables 'betas' (nr_betas x nr_signals) and 'residuals' (nr_signals x nr_samples) are created.
        """
        if alphas is None:
            alphas = np.logspace(7, 0, 20)
        self.rcv = linear_model.RidgeCV(alphas=alphas, 
                fit_intercept=False, 
                cv=cv) 
        self.rcv.fit(self.design_matrix.T, self.resampled_signal.T)

        self.betas = self.rcv.coef_.T
        self.residuals = self.resampled_signal - self.rcv.predict(self.design_matrix.T)

        self.logger.debug('performed ridge regression on %s design_matrix and %s signal, resulting alpha value is %f' % (str(self.design_matrix.shape), str(self.resampled_signal.shape), self.rcv.alpha_))

    def betas_for_cov(self, covariate = '0'):
        """betas_for_cov returns the beta values (i.e. IRF) associated with a specific covariate.

            :param covariate: name of covariate.
            :type covariate: string
        """
        # find the index in the designmatrix of the current covariate
        this_covariate_index = list(self.covariates.keys()).index(covariate)
        return self.betas[int(this_covariate_index*self.deconvolution_interval_size):int((this_covariate_index+1)*self.deconvolution_interval_size)]

    def betas_for_events(self):
        """betas_for_events creates an internal self.betas_per_event_type array, of (nr_covariates x self.devonvolution_interval_size), 
        which holds the outcome betas per event type,in the order generated by self.covariates.keys()
        """
        self.betas_per_event_type = np.zeros((len(self.covariates), self.deconvolution_interval_size, self.resampled_signal.shape[0]))
        for i, covariate in enumerate(self.covariates.keys()):
            self.betas_per_event_type[i] = self.betas_for_cov(covariate)

    def predict_from_design_matrix(self, design_matrix):
        """predict_from_design_matrix predicts signals given a design matrix.

            :param design_matrix: design matrix from which to predict a signal.
            :type design_matrix: numpy array, (nr_samples x betas.shape)
            :returns: predicted signal(s) 
            :rtype: numpy array (nr_signals x nr_samples)
        """
        # check if we have already run the regression - which is necessary
        assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
        assert design_matrix.shape[0] == self.betas.shape[0], \
                    'designmatrix needs to have the same number of regressors as the betas already calculated'

        # betas = np.copy(self.betas.T, order="F", dtype = np.float32)
        # f_design_matrix = np.copy(design_matrix, order = "F", dtype = np.float32)

        prediction = np.dot(self.betas.astype(np.float32).T, design_matrix.astype(np.float32))

        return prediction

    def calculate_rsq(self):
        """calculate_rsq calculates coefficient of determination, or r-squared, defined here as 1.0 - SS_res / SS_tot. rsq is only calculated for those timepoints in the data for which the design matrix is non-zero.
        """
        assert hasattr(self, 'betas'), 'no betas found, please run regression before rsq'

        explained_times = self.design_matrix.sum(axis = 0) != 0

        explained_signal = self.predict_from_design_matrix(self.design_matrix)
        self.rsq = 1.0 - np.sum((explained_signal[:,explained_times] - self.resampled_signal[:,explained_times])**2, axis = -1) / np.sum(self.resampled_signal[:,explained_times].squeeze()**2, axis = -1)
        self.ssr = np.sum((explained_signal[:,explained_times] - self.resampled_signal[:,explained_times])**2, axis = -1)
        return np.squeeze(self.rsq)

    def bootstrap_on_residuals(self, nr_repetitions = 1000):
        """bootstrap_on_residuals bootstraps, by shuffling the residuals. bootstrap_on_residuals should only be used on single-channel data, as otherwise the memory load might increase too much. This uses the lstsq backend regression for a single-pass fit across repetitions. Please note that shuffling the residuals may change the autocorrelation of the bootstrap samples relative to that of the original data and that may reduce its validity. Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals

            :param nr_repetitions: number of repetitions for the bootstrap.
            :type nr_repetitions: int

        """
        assert self.resampled_signal.shape[0] == 1, \
                    'signal input into bootstrap_on_residuals cannot contain signals from multiple channels at once, present shape %s' % str(self.resampled_signal.shape)
        assert hasattr(self, 'betas'), 'no betas found, please run regression before bootstrapping'

        # create bootstrap data by taking the residuals
        bootstrap_data = np.zeros((self.resampled_signal_size, nr_repetitions))
        explained_signal = self.predict_from_design_matrix(self.design_matrix).T

        for x in range(bootstrap_data.shape[-1]): # loop over bootstrapsamples
            bootstrap_data[:,x] = (self.residuals.T[np.random.permutation(self.resampled_signal_size)] + explained_signal).squeeze()

        self.bootstrap_betas, bs_residuals, rank, s = LA.lstsq(self.design_matrix.T, bootstrap_data)

        self.bootstrap_betas_per_event_type = np.zeros((len(self.covariates), self.deconvolution_interval_size, nr_repetitions))

        for i, covariate in enumerate(list(self.covariates.keys())):
            # find the index in the designmatrix of the current covariate
            this_covariate_index = list(self.covariates.keys()).index(covariate)
            self.bootstrap_betas_per_event_type[i] = self.bootstrap_betas[this_covariate_index*self.deconvolution_interval_size:(this_covariate_index+1)*self.deconvolution_interval_size]













