#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

from __future__ import division

import numpy as np
import scipy as sp

import matplotlib.pyplot as pl
# %matplotlib inline 
# %pylab osx

import seaborn as sn
sn.set(style="ticks")

from FIRDeconvolution import FIRDeconvolution

# signal parameters
signal_sample_frequency = 15
event_1_gain, event_2_gain = 2.3, 0.85
noise_gain = 0.75

# deconvolution parameters
deconv_sample_frequency = 3.0
deconvolution_interval = [-5, 25]

# how many time points to plot in figures
plot_time = 8000

# create some exponentially distributed random ISI events (Dale, 1999) of which we will create and deconvolve responses. 
period_durs = np.random.gamma(4.0,1.5,size = 600)
events = period_durs.cumsum()
events_1, events_2 = events[0::2], events[1::2]

durations_1, durations_2 = np.random.gamma(1.9,0.25,size = events_1.shape[0]), np.random.gamma(1.9,0.25,size = events_2.shape[0])

# these events are scaled with their own underlying covariate. 
# for instance, you could have a model-based variable that scales the signal on a per-trial basis. 
events_gains_1 = np.random.randn(len(events_1))*0.4
events_gains_2 = np.random.randn(len(events_2))*2.4

# We create an IRF, using a standard BOLD response.

def double_gamma_with_d(x, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, c = 0.35,d1=5.4,d2=10.8):
    return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])

hrf_1 = double_gamma_with_d(np.linspace(0,25,25*signal_sample_frequency), a1 = 4.5, a2 = 10, d1 = 5.0, d2 = 10.0)
hrf_2 = double_gamma_with_d(np.linspace(0,25,25*signal_sample_frequency), a1 = 1.5, a2 = 10, d1 = 3.0, d2 = 10.0)
# hrf = hrf/np.abs(hrf).sum()

f = pl.figure(figsize = (10,4))
pl.plot(np.linspace(0,25,25*signal_sample_frequency), hrf_1, 'g')
pl.plot(np.linspace(0,25,25*signal_sample_frequency), hrf_2, 'b')
pl.axhline(0, lw=0.5, color = 'k')
sn.despine()

# Using this IRF we're going to create two signals
# signal gains are determined by random covariate and a standard gain
# we mix them all together with some noise, injected on the signal, not the events.

times = np.arange(0,events.max()+45.0,1.0/signal_sample_frequency)

event_1_in_times = np.array([((times>te) * (times<te+d)) * event_1_gain for te, d in zip(events_1, durations_1)]).sum(axis = 0)
event_2_in_times = np.array([((times>te) * (times<te+d)) * event_2_gain for te, d in zip(events_2, durations_2)]).sum(axis = 0)

signal_1 = sp.signal.fftconvolve(event_1_in_times, hrf_1, 'full')[:times.shape[0]]
signal_2 = sp.signal.fftconvolve(event_2_in_times, hrf_2, 'full')[:times.shape[0]]

# combine the two signals with one another, z-score and add noise
input_data = signal_1 + signal_2
input_data = (input_data - np.mean(input_data)) / input_data.std()
input_data += np.random.randn(input_data.shape[0]) * noise_gain



f = pl.figure(figsize = (10,8))
s = f.add_subplot(311)
pl.plot(np.arange(plot_time), event_1_in_times[:plot_time], 'b-')
pl.plot(np.arange(plot_time), event_2_in_times[:plot_time], 'g-')
sn.despine()

pl.axhline(0, lw=0.5, color = 'k')
s = f.add_subplot(312)

pl.plot(np.arange(plot_time), signal_1[:plot_time], 'b--')
pl.plot(np.arange(plot_time), signal_2[:plot_time], 'g--')
sn.despine()

pl.axhline(0, lw=0.5, color = 'k')
s = f.add_subplot(313)

pl.plot(np.arange(plot_time), input_data[:plot_time], 'k-')
pl.legend(['events_1', 'events_2', 'signal_1', 'signal_2', 'total signal'])
pl.axhline(0, lw=0.5, color = 'k')

sn.despine()
# pl.tight_layout()


# Up until now, we just created data. 
# Now, we'll use the actual deconvolution package.

# first, we initialize the object
fd = FIRDeconvolution(
            signal = input_data, 
            events = [events_1, events_2], 
            event_names = ['event_1', 'event_2'], 
            sample_frequency = signal_sample_frequency,
            deconvolution_frequency = deconv_sample_frequency,
            deconvolution_interval = deconvolution_interval
            )

# we then tell it to create its design matrix
fd.create_design_matrix()

# perform the actual regression, in this case with the statsmodels backend
fd.regress(method = 'lstsq')

# and partition the resulting betas according to the different event types
fd.betas_for_events()

fd.calculate_rsq()

# and we see what we've done

f = pl.figure(figsize = (10,8))
s = f.add_subplot(311)
s.set_title('FIR responses, Rsq is %1.3f'%fd.rsq)
for dec in fd.betas_per_event_type.squeeze():
    pl.plot(fd.deconvolution_interval_timepoints, dec)
# fd.covariates, being a dictionary, cannot be assumed to maintain the event order. 
# working on a fix here....
pl.legend(fd.covariates.keys())
sn.despine()

pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
s = f.add_subplot(312)
s.set_title('design matrix')
pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, cmap = 'RdBu', interpolation = 'nearest', rasterized = True)
sn.despine()

s = f.add_subplot(313)
s.set_title('data and predictions')
pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
        fd.resampled_signal[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)].T, 'r')
pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
        fd.predict_from_design_matrix(fd.design_matrix[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)]).T, 'k')
pl.legend(['signal','explained'])
sn.despine()
# pl.tight_layout()




# Now add durations

# first, we initialize the object
fd = FIRDeconvolution(
            signal = input_data, 
            events = [events_1, events_2], 
            event_names = ['event_1', 'event_2'], 
            durations = {'event_1': durations_1, 'event_2': durations_2},
            sample_frequency = signal_sample_frequency,
            deconvolution_frequency = deconv_sample_frequency,
            deconvolution_interval = deconvolution_interval
            )

# we then tell it to create its design matrix
fd.create_design_matrix()

# perform the actual regression, in this case with the statsmodels backend
fd.regress(method = 'lstsq')

# and partition the resulting betas according to the different event types
fd.betas_for_events()

fd.calculate_rsq()

# and we see what we've done

f = pl.figure(figsize = (10,8))
s = f.add_subplot(311)
s.set_title('FIR responses, Rsq is %1.3f'%fd.rsq)
for dec in fd.betas_per_event_type.squeeze():
    pl.plot(fd.deconvolution_interval_timepoints, dec)
# fd.covariates, being a dictionary, cannot be assumed to maintain the event order. 
# working on a fix here....
pl.legend(fd.covariates.keys())
sn.despine()

pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
s = f.add_subplot(312)
s.set_title('design matrix')
pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, cmap = 'RdBu', interpolation = 'nearest', rasterized = True)
sn.despine()

s = f.add_subplot(313)
s.set_title('data and predictions')
pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
        fd.resampled_signal[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)].T, 'r')
pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
        fd.predict_from_design_matrix(fd.design_matrix[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)]).T, 'k')
pl.legend(['signal','explained'])
sn.despine()
# pl.tight_layout()











