#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:09:54 2025

@author: cindy
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt 
warnings.filterwarnings("ignore")

os.chdir('/mnt/storage/HRF_deconvolution')
path = os.getcwd() + "/tvb-root-amogh/scientific_library"
sys.path.insert(0, path)

from tvb.datatypes import equations
from tvb.simulator import monitors


import numpy as np
from rsHRF.spm_dep import spm as spm
import matplotlib.pyplot as plt
from scipy import stats


file_path = '/mnt/storage/neuronumba/demos/2_MIST_parcellations/031-MIST/Data_Raw_031-MIST/sub-101309/rfMRI_REST1_LR_BOLD.csv'
roiTimeSeries = np.loadtxt(file_path)
print("Shape of the BOLD time-series: ", roiTimeSeries.shape)
# Shape of the BOLD time-series:  (1200, 31)

bold_rsHRF_roiTS = monitors.Bold(hrf_kernel = equations.RestingStateHRF(roiTS = roiTimeSeries))


hrf_params = {
    "estimation": 'canon2dd', 
    "passband": [0.01, 0.08], 
    "TR": 0.72,  # Use your actual TR 
    "T": 3, 
    "T0": 1, 
    "TD_DD": 2, 
    "AR_lag": 1, 
    "thr": 1, 
    "len": 24,  # Your HRF length 
    "min_onset_search": 3, 
    "max_onset_search": 8, 
    'localK': 1,
    "pjobs": 1  # Explicitly set parallel jobs
}

'''   equations.py(): RestingStateHRF.evaluate()   '''

from rsHRF import processing, canon, sFIR, utils, basis_functions #changed         

para = hrf_params
pjobs = hrf_params['pjobs']
para['dt'] = para['TR'] / para['T'] # 0.24 because 0.72s / 3 
para['lag'] = np.arange(np.fix(para['min_onset_search'] / para['dt']),    
                    np.fix(para['max_onset_search'] / para['dt']) + 1,
                    dtype='int') # an array of (22,) from [12 13 14 .... 32 33] # min to max seconds: ie 3 to 8 seconds: 5 seconds but at timesteps of 0.24s
bold_sig = bold_rsHRF_roiTS.hrf_kernel.roiTS                  

# emperical region-wise BOLD response
bold_sig = stats.zscore(bold_sig, ddof=1)                                       # normalizing the BOLD time-series
bold_sig = np.nan_to_num(bold_sig)                                           # removing nan values

# bold_sig = processing.rest_filter.rest_IdealFilter(bold_sig, para['TR'], para['passband'])                # applying the band-pass filter

# bold_sig_shape = bold_sig.shape #(1200, 31)
# bold_sig_filt_shape = bold_sig_filt.shape

# for estimating bf: just uses bf shape not the actual content so it's okay
bf = basis_functions.basis_functions.get_basis_function(bold_sig.shape, para)  
# should this be per person? not sure of variability/stability here

%matplotlib inline

# orthogonalized
plt.plot(bf)

# regular, without filtering at all
para['temporal_mask'] = []
beta_hrf, event_bold = utils.hrf_estimation.compute_hrf(bold_sig, para, para['temporal_mask'], para['pjobs'], bf=bf)
# returns: (5, 31)

hrfa = np.dot(bf, beta_hrf[np.arange(0, bf.shape[1]), :]) # (101, 31) # creates the region-specific HRF as a combination of the basis functions, weighted by the corresponding betas
plt.plot(hrfa)

# assess optimal time lags 
# plt.hist(beta_hrf[4,:])

delays = beta_hrf[4,:]  # Or whichever index contains the delays
regions = np.arange(len(delays))

plt.figure(figsize=(12, 6))
plt.bar(regions, delays)
plt.xlabel('Region Index')
plt.ylabel('Optimal Delay (time steps)')
plt.title('Optimal Delays by Brain Region')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# assess betas
# Assuming beta_hrf[:3,:] contains the weights for the three basis functions
weights = beta_hrf[:3,:]  # First 3 rows are the weights

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(weights.shape[1])  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['Canonical', 'Time Derivative', 'Dispersion Derivative']
for i in range(3):
    ax.bar(x + (i-1)*width, weights[i,:], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Weight Value')
ax.set_title('Basis Function Weights by Region')
ax.set_xticks(x[::10])  # Show every 10th region
ax.legend()
plt.tight_layout()
plt.show()


# assess event numbers 
event_numbers = []
for i in event_bold:
    event_numbers.append(len(i))
    
regions = np.arange(len(event_numbers))

plt.figure(figsize=(12, 6))
plt.bar(regions, event_numbers)
plt.xlabel('Region Index')
plt.ylabel('Number of Events Detected')
plt.title('Event Number Detected by Brain Region')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Assuming intercept is stored at index 3
intercepts = beta_hrf[3,:]

plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(intercepts)), intercepts)
plt.xlabel('Region Index')
plt.ylabel('Intercept Value')
plt.title('HRF Intercept by Brain Region')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% 
# without changing bf
# edited hrf_estimation script to use filtered bold_sig for bold event detection then unfiltered bold_sig to fit the HRFs and estimate betas
bf 

# edited hrf_estimation script to use filtered bold_sig for bold event detection then unfiltered bold_sig to fit the HRFs and estimate betas
para['temporal_mask'] = []
beta_hrf_filt, event_bold_filt = utils.hrf_estimation.compute_hrf(bold_sig, para, para['temporal_mask'], para['pjobs'], bf=bf)

hrfa_filt = np.dot(bf, beta_hrf_filt[np.arange(0, bf.shape[1]), :])
plt.plot(hrfa_filt)


# assess optimal time lags 
# plt.hist(beta_hrf[4,:])

delays = beta_hrf_filt[4,:]  # Or whichever index contains the delays
regions = np.arange(len(delays))

plt.figure(figsize=(12, 6))
plt.bar(regions, delays)
plt.xlabel('Region Index')
plt.ylabel('Optimal Delay (time steps)')
plt.title('Optimal Delays by Brain Region - Filtered')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# assess betas
# Assuming beta_hrf[:3,:] contains the weights for the three basis functions
weights = beta_hrf_filt[:3,:]  # First 3 rows are the weights

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(weights.shape[1])  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['Canonical', 'Time Derivative', 'Dispersion Derivative']
for i in range(3):
    ax.bar(x + (i-1)*width, weights[i,:], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Weight Value')
ax.set_title('Basis Function Weights by Region - Filtered')
ax.set_xticks(x[::10])  # Show every 10th region
ax.legend()
plt.tight_layout()
plt.show()


# assess event numbers 
event_numbers = []
for i in event_bold_filt:
    event_numbers.append(len(i))
    
regions = np.arange(len(event_numbers))

plt.figure(figsize=(12, 6))
plt.bar(regions, event_numbers)
plt.xlabel('Region Index')
plt.ylabel('Number of Events Detected')
plt.title('Event Number Detected by Brain Region - Filtered')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Assuming intercept is stored at index 3
intercepts = beta_hrf_filt[3,:]

plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(intercepts)), intercepts)
plt.xlabel('Region Index')
plt.ylabel('Intercept Value')
plt.title('HRF Intercept by Brain Region - Filtered')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% what is going on with time betas?? 

time_betas = [beta_hrf[1,:], beta_hrf_filt[1,:]] 

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(time_betas[0]))  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['nonfiltered', 'filtered']
for i in range(2):
    ax.bar(x + (i-1)*width, time_betas[i], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Weight Value')
ax.set_title('Comparing Temporal Betas - nonfiltered vs filtered')
ax.legend()
plt.tight_layout()
plt.show()


canon_betas = [beta_hrf[0,:], beta_hrf_filt[0,:]] 

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(canon_betas[0]))  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['nonfiltered', 'filtered']
for i in range(2):
    ax.bar(x + (i-1)*width, canon_betas[i], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Weight Value')
ax.set_title('Comparing Canon Betas - nonfiltered vs filtered')
ax.legend()
plt.tight_layout()
plt.show()

dispersion_betas = [beta_hrf[2,:], beta_hrf_filt[2,:]] 

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(dispersion_betas[0]))  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['nonfiltered', 'filtered']
for i in range(2):
    ax.bar(x + (i-1)*width, dispersion_betas[i], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Weight Value')
ax.set_title('Comparing Dispersion Betas - nonfiltered vs filtered')
ax.legend()
plt.tight_layout()
plt.show()

time_lags = [beta_hrf[4,:], beta_hrf_filt[4,:]] 

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(time_lags[0]))  # Region indices
width = 0.25  # Width of the bars

# Plot bars for each basis function
basis_names = ['nonfiltered', 'filtered']
for i in range(2):
    ax.bar(x + (i-1)*width, time_lags[i], width, label=basis_names[i])

ax.set_xlabel('Region Index')
ax.set_ylabel('Optimal Time Lag (time bins)')
ax.set_title('Comparing Optimal Time Lags - nonfiltered vs filtered')
ax.legend()
plt.tight_layout()
plt.show()


#%% 

bds = b.compute_bold(signal, sampling_period)