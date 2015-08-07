

#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
from __future__ import division

import unittest
import logging

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl

# import pandas as pd
import statsmodels.api as sm
# import sklearn.linear_model.Ridge as Ridge
import numpy.linalg as LA

# import bottleneck as bn
import seaborn as sn

sn.set(style="ticks")	

from IPython import embed as shell


class FIRDeconvolution(object): 
	"""Instances of FIRDeconvolutionOperator can be used to perform FIR fitting on time-courses."""

	def __init__(self, signal, events, event_names = [], covariates = None, sample_frequency = 1.0, deconvolution_interval = [-0.5, 5], deconvolution_frequency = None):
		"""
		FIRDeconvolution takes a signal (signals X nr samples), sampled at sample_frequency in Hz, and deconvolves this signal using least-squares FIR fitting. 
		The resulting FIR curves are sampled at deconvolution_frequency in Hz, for the interval deconvolution_interval in [start, end] seconds.
		Event occurrence times are given in seconds.
		covariates is a dictionary, with keys starting with the event they should be 'attached' to, followed by a _ sign and further name. 
		The values of the covariate dictionary are numpy arrays with the same length as the original events.
		"""


		self.logger = logging.getLogger('FIRDeconvolution')
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		self.logger.addHandler(ch)

		self.logger.info('initializing deconvolution with signal sample freq %2.2f, etc etc.' % (sample_frequency))

		self.signal = signal 
		if len(self.signal.shape) == 1:
			self.signal = self.signal[np.newaxis, :]

		self.events = events 
		if event_names == []:
			self.event_names = [str(i) for i in np.arange(self.events.shape[0])] 
		else:
			self.event_names = event_names
		assert len(self.event_names) == len(self.events), \
					'number of event names (%i, %s) does not align with number of event definitions (%i)' %(len(self.event_names), self.event_names, len(self.events))

		self.events = {}
		for ev, evn in zip(events, self.event_names):
			self.events.update({evn: ev})

		# if no covariates, we make a new covariate dictionary specifying only ones.
		# we will loop over these covariates instead of the event list themselves to create design matrices
		if covariates == None:
			self.covariates = {}
			for ev, evn in zip(self.events, self.event_names):
				self.covariates.update({evn: np.ones(len(self.events[ev]))})
		else:
			self.covariates = covariates

		# if we did not create the covariates dictionary, this could fail
		nr_events_per_event_type = [len(self.events[e]) for e in self.events]
		nr_covariates_per_event_type = [len(self.covariates[c]) for c in self.covariates]
		# assert nr_covariates_per_event_type == nr_events_per_event_type, \
		# 	'numbers of events and covariates don\'t line up.\n%s svs %s' % ( str(nr_covariates_per_event_type), str(nr_events_per_event_type) )
		self.number_of_event_types = len(self.covariates)

		self.sample_frequency = sample_frequency
		self.deconvolution_interval = deconvolution_interval
		if deconvolution_frequency is None:
			self.deconvolution_frequency = sample_frequency
		else:
			self.deconvolution_frequency = deconvolution_frequency

		# first checks
		self.resampling_factor = self.sample_frequency/self.deconvolution_frequency
		assert round(self.sample_frequency/self.deconvolution_frequency) == self.sample_frequency/self.deconvolution_frequency, \
				'sample frequency and deconvolution frequency should be relative integers'
		
		self.deconvolution_interval_size = (self.deconvolution_interval[1] - self.deconvolution_interval[0]) * self.deconvolution_frequency
		assert round(self.deconvolution_interval_size) == self.deconvolution_interval_size, 'self.sample_interval_size should be integer'
		self.deconvolution_interval_timepoints = np.linspace(self.deconvolution_interval[0],self.deconvolution_interval[1],self.deconvolution_interval_size)

		##
		#	create instance variables that determine calculations
		##

		# duration of signal in seconds and at deconvolution frequency
		self.signal_duration = self.signal.shape[-1] / self.sample_frequency
		self.resampled_signal_size = self.signal_duration*self.deconvolution_frequency
		self.resampled_signal = sp.signal.resample(self.signal, self.resampled_signal_size, axis = -1)

		# indices of events in the resampled signal, keeping this as a list instead of an array
		# at this point we take into account the offset encoded in self.deconvolution_interval[0]
		self.event_times_indices = {}
		for ev in self.events:
			self.event_times_indices.update({ev: np.array((self.events[ev] + self.deconvolution_interval[0]) * self.deconvolution_frequency, dtype = np.int)})

	def create_event_regressors(self, event_times_indices, covariates = None):
		"""
		create_event_regressors takes the index of the event for which to create the regressors. 
		it may or may not be supplied with a set of covariates for these events.
		"""

		# check covariates
		if covariates is None:
			covariates = np.ones(self.event_times_indices)

		# set up output array
		regressors_for_event = np.zeros((self.deconvolution_interval_size, self.resampled_signal_size))

		# fill up output array
		for cov, eti in zip(covariates, event_times_indices):
			self.logger.info('deconv samples are starting before the data starts.')
			self.logger.info('deconv samples are continuing after the data stops.')
			self.logger.info('event falls outside of the scope of the data.')
			# assert eti > 0, \
			# 		'deconv samples are starting before the data ends.'
			# assert eti+self.deconvolution_interval_size < self.resampled_signal_size, \
			# 		'deconv samples are continuing after the data stops.'
			# assert eti < self.resampled_signal_size, \
			# 		'event falls outside of the scope of the data.'
			time_range = np.arange(max(0,eti),min(eti+self.deconvolution_interval_size, self.resampled_signal_size), dtype = int)
			if len(time_range) > 0: # only incorporate sensible events.
				regressors_for_event[-len(time_range):,time_range] += (np.diag(np.ones(self.deconvolution_interval_size)) * cov)[-len(time_range):,-len(time_range):]
		
		return regressors_for_event

	def create_design_matrix(self):
		"""
		create_design_matrix calls create_event_regressors for each of the covariates in the self.covariates dict. 
		self.designmatrix is created and is shaped (nr_regressors, self.resampled_signal.shape[-1])
		"""

		self.design_matrix = np.zeros((self.number_of_event_types*self.deconvolution_interval_size, self.resampled_signal_size))

		for i, covariate in enumerate(self.covariates.keys()):
			# document the creation of the designmatrix step by step
			self.logger.info('creating regressor for ' + covariate)
			indices = np.arange(i*self.deconvolution_interval_size,(i+1)*self.deconvolution_interval_size, dtype = int)
			if len(covariate.split('.')) > 0:
				which_event_time_indices = covariate.split('.')[0]
			else:
				which_event_time_indices = covariate
			self.design_matrix[indices] = self.create_event_regressors(self.event_times_indices[which_event_time_indices], self.covariates[covariate])

	def add_continuous_regressors_to_design_matrix(self, regressor):
		"""
		add_continuous_regressors_to_design_matrix expects as input a regressor shaped similarly to the design matrix.
		"""

		if len(regressors.shape) == 1:
			regressors = regressor[np.newaxis, :]
		assert regressors.shape[1] is self.resampled_signal.shape[1], \
				'additional regressor shape %s does not conform to designmatrix shape %s' % (regressors.shape, self.resampled_signal.shape)
		# and, an hstack append
		self.design_matrix = np.hstack((self.design_matrix, regressors))

	def regress(self, method = 'lstsq'):
		"""
		regress performs linear least squares regression of the designmatrix on the data. 
		one may choose a method out of the options 'lstsq', 'sm_ols'.
		"""
		# shell()

		if method is 'lstsq':
			betas, residuals, rank, s = LA.lstsq(self.design_matrix.T, self.resampled_signal.T)
		elif method is 'sm_ols':
			assert self.resampled_signal.shape[1] == 1, \
					'signal input into statsmodels OLS cannot contain multiple signals at once, present shape %s' % str(self.resampled_signal.shape)
			model = sm.OLS(self.resampled_signal,self.design_matrix.T)
			results = model.fit()
			betas = results.params
			residuals = model.resid

		self.betas = betas
		self.residuals = residuals

	def betas_for_cov(self, covariate = '0'):
		"""
		betas_for_cov returns the betas associated with a specific covariate.
		covariate is specified by name.
		"""

		# find the index in the designmatrix of the current covariat
		this_covariate_index = self.covariates.keys().index(covariate)
		return self.betas[this_covariate_index*self.deconvolution_interval_size:(this_covariate_index+1)*self.deconvolution_interval_size]

	def betas_for_events(self):
		"""
		betas_for_events creates an internal self.betas_per_event_type array, of (nr_covariates x self.devonvolution_interval_size)
		"""

		self.betas_per_event_type = np.zeros((len(self.covariates), self.deconvolution_interval_size, self.resampled_signal.shape[0]))
		for i, covariate in enumerate(self.covariates.keys()):
			self.betas_per_event_type[i] = self.betas_for_cov(covariate)

	def predict_from_design_matrix(self, design_matrix):
		"""
		predict_from_design_matrix takes a design matrix (timepoints, betas.shape)
		"""
		# check if we have already run the regression - which is necessary
		assert hasattr(self, 'betas'), 'no betas found, please run regression before prediction'
		assert design_matrix.shape[0] == self.betas.shape[0], \
					'designmatrix needs to have the same number of regressors as the betas already calculated'
		return np.dot(self.betas.T, design_matrix)


