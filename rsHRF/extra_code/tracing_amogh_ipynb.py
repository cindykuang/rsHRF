#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:34:55 2025

following Amogh Johri's rsHRF github .ipynb manual

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

# import sys
# sys.path.append('/absolute/path/to/cloned/tvb-repo')
# # Now you can import from the repo
# import tvb.module

from tvb.datatypes import equations
from tvb.simulator import monitors

#default BOLD monitor


bold_default = monitors.Bold()
bold_default.configure()
bold_default

#vars:
''' vars(bold_default)
Out[8]: 
{'gid': UUID('0545ddb0-a370-40b8-a53d-8691ec52b526'),
  'title': 'Bold gid: 0545ddb0-a370-40b8-a53d-8691ec52b526',
  'tags': {},
  'log': <Logger tvb.simulator.monitors (INFO)>,
  'period': 2000.0,
  'hrf_kernel': <tvb.datatypes.equations.FirstOrderVolterra at 0x7aef553b8100>,
  'hrf_length': 20000.0,
  'variables_of_interest': None}
'''


# In general, the sampling period of a monitor is in milliseconds and must be an integral multiple of the integration-step size used in a simulation.
# my dt was 0.09 (integration step size used in the simulation) 
# so my sampling size for the monitor was 0.72ms 
# BOLD needs to downsample it to 0.72s

bold_default.dt = 0.09

bold_default.compute_hrf()
hrf_default  = bold_default.hemodynamic_response_function
print("Shape of the hrf is: ", hrf_default.shape)

# Shape of the hrf is:  (1, 5000)
# The shape of the hrf is 1x5000, where 5000 corresponds to the dimension along signal's x-axis. 

# plot the kernel
%matplotlib inline
plt.plot(bold_default._stock_time, hrf_default.T[::-1]); 
plt.ylabel('hrf');
plt.xlabel('time [sec]')
# plot the maximum
plt.plot(bold_default._stock_time[hrf_default.T[::-1].argmax()], hrf_default.T[::-1].max(), 'ko')

'''
NOW TRYING ON RSHRF 
'''

# need ROI timeseries (empirical BOLD)
# let's do 

file_path = '/mnt/storage/neuronumba/demos/2_MIST_parcellations/031-MIST/Data_Raw_031-MIST/sub-101309/rfMRI_REST1_LR_BOLD.csv'
roiTimeSeries = np.loadtxt(file_path)
print("Shape of the BOLD time-series: ", roiTimeSeries.shape)
# Shape of the BOLD time-series:  (1200, 56)

# plotting the time-series corresponding to the first-region
plt.plot(roiTimeSeries.T[0])
plt.ylabel('BOLD');
plt.xlabel('time [sec]')

# configure 

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
#parameters=hrf_params, HRF_length = 24.0, TR = 720)
bold_rsHRF_roiTS = monitors.Bold(hrf_kernel = equations.RestingStateHRF(roiTS = roiTimeSeries))
bold_rsHRF_roiTS.dt = 0.72 #ms, because this is coming out of temporal average monitor
bold_rsHRF_roiTS.period = 720
bold_rsHRF_roiTS.hrf_length = 24000
bold_rsHRF_roiTS.configure()
bold_rsHRF_roiTS

''' vars(bold_rsHRF_roiTS)
{'gid': UUID('e250d119-7af0-43b4-a3cc-8a4abd8fefa8'),
 'title': 'Bold gid: e250d119-7af0-43b4-a3cc-8a4abd8fefa8',
 'hrf_kernel': <tvb.datatypes.equations.RestingStateHRF at 0x7aef53bcb430>,
 'tags': {},
 'log': <Logger tvb.simulator.monitors (INFO)>,
 'dt': 0.72,
 'period': 720.0,
 'hrf_length': 20000.0,
 'variables_of_interest': None}
'''

bold_rsHRF_roiTS.compute_hrf()


#%%

def precompute_hrf_and_save(bold_timeseries, output_filename, params):
    """
    Precompute the HRF for each region and save to a file
    This avoids the issues with inhomogeneous shapes by handling them before TVB needs the data
    """
    import numpy as np
    from scipy import stats
    from rsHRF import processing, basis_functions, utils
    
    # Preprocess BOLD signal
    bold_sig = stats.zscore(bold_timeseries, ddof=1)
    bold_sig = np.nan_to_num(bold_sig)
    bold_sig = processing.rest_filter.rest_IdealFilter(bold_sig, params['TR'], params['passband'])
    
    # Configure parameters
    params['dt'] = params['TR'] / params['T']
    params['lag'] = np.arange(np.fix(params['min_onset_search'] / params['dt']),
                            np.fix(params['max_onset_search'] / params['dt']) + 1,
                            dtype='int')
    
    # Get basis functions
    bf = basis_functions.basis_functions.get_basis_function(bold_sig.shape, params)
    
    # Process each region separately
    N, nvar = bold_sig.shape
    hrfa_list = []
    
    for i in range(nvar):
        print(f"Processing region {i+1}/{nvar}")
        try:
            # Try to estimate HRF for just this region
            region_data = bold_sig[:, i:i+1]
            region_beta, _ = utils.hrf_estimation.compute_hrf(
                region_data, params, [], params['pjobs'], bf=bf)
                
            # Convert beta to HRF
            region_hrfa = np.dot(bf, region_beta[np.arange(0, bf.shape[1]), :])
            hrfa_list.append(region_hrfa.flatten())
        except Exception as e:
            print(f"Error processing region {i}: {e}")
            # Use canonical HRF as fallback
            # default_hrf = np.zeros(bf.shape[0])
            # default_hrf[5:15] = [0.1, 0.3, 0.5, 0.8, 1.0, 0.9, 0.6, 0.4, 0.2, 0.1]  # Simplified HRF shape
            # hrfa_list.append(default_hrf)
    
    # Save to file
    hrf_data = np.vstack(hrfa_list)
    np.savetxt(output_filename, hrf_data)
    print(f"HRF data saved to {output_filename}")
    
    return hrf_data

filename = '/mnt/storage/HRF_deconvolution/MIST-031_sub-101309_REST1_LR_precomputed_HRFs.txt'
hrf_data = precompute_hrf_and_save(roiTimeSeries, filename, hrf_params)

plt.figure(figsize=(12, 6))
for i in range(hrf_data.shape[0]):
    plt.plot(hrf_data[i], label=f'Region {i+1}')
plt.legend()
plt.title('Sample HRFs')
plt.xlabel('Time (samples)')
plt.ylabel('HRF Amplitude')
plt.show()

#%% tracing equations.py

file_path = '/mnt/storage/neuronumba/demos/2_MIST_parcellations/031-MIST/Data_Raw_031-MIST/sub-101309/rfMRI_REST1_LR_BOLD.csv'
roiTimeSeries = np.loadtxt(file_path)
print("Shape of the BOLD time-series: ", roiTimeSeries.shape)
# Shape of the BOLD time-series:  (1200, 31)

bold_rsHRF_roiTS = monitors.Bold(hrf_kernel = equations.RestingStateHRF(roiTS = roiTimeSeries))
#  'hrf_kernel': <tvb.datatypes.equations.RestingStateHRF at 0x79c8f3fc18a0>,


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
bold_sig = processing.rest_filter.rest_IdealFilter(bold_sig, para['TR'], para['passband'])                # applying the band-pass filter
bold_sig_shape = bold_sig.shape #(1200, 31)

'''looking at pass band filter'''

# def rest_IdealFilter(x, TR, Bands, m=5000):
#     nvar = x.shape[1] #31 (nROIs)
#     nbin = int(np.ceil(nvar/m)) # 1
#     for i in range(1, nbin + 1): # for i in range (1, 2): # it's just going to be 1
#         if i != nbin:
#             ind_X = [j for j in range((i-1)*m, i*m)] # range(0, 5000)
#         else:
#             ind_X = [j for j in range((i-1)*m, nvar)] # range(0, 31)
#         x1 = x[:, ind_X]
#         x1 = conn_filter(TR,Bands,x1) + np.matlib.repmat(np.mean(x1), x1.shape[0], 1)
#         x[:,ind_X] = x1
#     return x

# x = bold_sig
# para['TR'] = para['TR']
# para['']

''' bf = basis_functions.basis_functions.get_basis_function(bold_sig.shape, para) '''

N, nvar = bold_sig_shape

''' bf = canon2dd_bf(bold_sig_shape, para) ''' 

''' bf = canon.canon_hrf2dd.wgr_spm_get_canonhrf(para) '''

'''    wgr_spm_get_canonhrf(xBF)   '''

import numpy as np
from rsHRF.spm_dep import spm as spm
import matplotlib.pyplot as plt

def spm_hrf(RT, P=None, fMRI_T=16):
    """
    @RT - scan repeat time
    @P  - parameters of the response function (two gamma functions)

    defaults  (seconds)
    %	P[0] - Delay of Response (relative to onset)	    6
    %	P[1] - Delay of Undershoot (relative to onset)     16
    %	P[2] - Dispersion of Response			            1
    %	P[3] - Dispersion of Undershoot			            1
    %	P[4] - Ratio of Response to Undershoot		        6
    %	P[5] - Onset (seconds)				                0
    %	P[6] - Length of Kernel (seconds)		           32

    hrf  - hemodynamic response function
    P    - parameters of the response function
    """
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    if P is not None:
        p[0:len(P)] = P
    _spm_Gpdf = lambda x, h, l: \
        np.exp(h * np.log(l) + (h - 1) * np.log(x) - (l * x) - gammaln(h))
    # modelled hemodynamic response function - {mixture of Gammas}
    dt = RT / float(fMRI_T) # this becomes 0.08? because 0.24 microtime res / float(3) again? 
    
    # time vector u represents 0s to 24s in timesteps of 0.08s (ranges from 0 to 300)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt # array of (301,) [0, 1, 2, 3 .... 398, 399, 400] 
    
    with np.errstate(divide='ignore'):  # Known division-by-zero
        # the HRF is computed as a difference of two gamma functions at each of the 301 timepoints (represented in time vector u)
        hrf = _spm_Gpdf(
            u, p[0] / p[2], dt / p[2]
        ) - _spm_Gpdf(
            u, p[1] / p[3], dt / p[3]
        ) / p[4]
            
    # downsampling the HRF 
    
    #[0:floor(p(7)/RT)] creates a vector [0,1,2,...,100]
    # Multiplying by fMRI_T (which is 3) gives [0,3,6,...,300]
    # Adding 1 gives [1,4,7,...,301]
    # Using these indices to subset hrf gives us 101 time points
    idx = np.arange(0, int((p[6] / RT) + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = np.nan_to_num(hrf)
    
    # normalized HRF to sum to 1 (why ?)
    hrf = hrf / np.sum(hrf)
    return hrf

dt     = para['dt'] #0.24 microtime resolution (finer than TR)
fMRI_T = para['T'] #3
p      = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
p[len(p) - 1] = para['len'] # just adjusting length to the set length at end of p array (replacing default 32s)
bf = spm.spm_hrf(dt, p, fMRI_T) # gave a (101,) array

# is dt input here supposed to be TR time (0.72s) or already microtime res (0.72/3 = 0.24s)???
# maybe dt=0.72s/T=3 in initial param settings in evaluate() in equations.py was a MISTAKE
# spm.spm_hrf expects its first parameter (RT) to be the TR of the empirical BOLD sampling rate (0.72s not 0.24s)
# this is timestep for creating basis functions for HRF- no inherent time res constraint - can have a smoother model curve to fit to the data or coarser

# if I try spm.spm_hrf(dt=0.72, p, fMRI_T) it gives (45,) - shape is the same approx but not as smoothed out

bf = bf[:, np.newaxis] # just adds an axis, makes it (101, 1)
if para['TD_DD']:
    dp   = 1
    p[5] = p[5] + dp # 0 + 1 = 1
    D    = (bf[:, 0] - spm.spm_hrf(dt, p, fMRI_T)) / dp # gives (101,)
    D    = D[:, np.newaxis] #(101, 1)
    bf   = np.append(bf, D, axis=1) #(101, 2)
    p[5] = p[5] - dp #reset p[5] back to 0

    if xBF['TD_DD'] == 2:
        dp   = 0.01
        p[2] = p[2] + dp # 1 + 0.01 = 1.01
        D    = (bf[:, 0] - spm.spm_hrf(dt, p, fMRI_T)) / dp #(101,)
        D    = D[:, np.newaxis] #(101,1)
        bf   = np.append(bf, D, axis=1) #(101, 3)
        

# so now bf is a matrix of (101, 3): each column is a basis function: canonical HRF, time derivative, dispersion derivative
# each was calculated at an ultra fine timestep of 0.08s
# time vector u represents 0s to 24s in timesteps of 0.08s (ranges from 0 to 300)
# the HRF is computed as a difference of two gamma functions at each of the 301 timepoints (represented in time vector u)
# then downsampled back down to: 101 timepoints, representing the HRF from 0s to 24s in steps of 0.24s

def wgr_spm_get_canonhrf(xBF):
    dt     = xBF['dt']
    fMRI_T = xBF['T']
    p      = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    p[len(p) - 1] = xBF['len']
    bf = spm.spm_hrf(dt, p, fMRI_T)
    bf = bf[:, np.newaxis]
    # time-derivative
    if xBF['TD_DD']:
        dp   = 1
        p[5] = p[5] + dp
        D    = (bf[:, 0] - spm.spm_hrf(dt, p, fMRI_T)) / dp
        D    = D[:, np.newaxis]
        bf   = np.append(bf, D, axis=1)
        p[5] = p[5] - dp
        # dispersion-derivative
        if xBF['TD_DD'] == 2:
            dp   = 0.01
            p[2] = p[2] + dp
            D    = (bf[:, 0] - spm.spm_hrf(dt, p, fMRI_T)) / dp
            D    = D[:, np.newaxis]
            bf   = np.append(bf, D, axis=1)
    return bf

%matplotlib inline
plt.plot(bf) #3 basis functions

''' BACK UP TO: bf = canon2dd_bf(bold_sig_shape, para) ''' 

# something about Volterra??

''' BACK UP TO: bf = basis_functions.basis_functions.get_basis_function(bold_sig.shape, para) '''

bf = spm.spm_orth(np.asarray(bf))
    
# Orthogonalization ensures that the basis functions are linearly independent and have zero correlation with each other
# Mathematically, it means that the dot product between any two basis functions equals zero:
# Preventing Collinearity: Without orthogonalization, the basis functions would be correlated, leading to collinearity issues during model fitting.
# Clearer Interpretation: Each basis function captures a distinct aspect of HRF variability, making the coefficients more interpretable.

''' beta_hrf, event_bold = utils.hrf_estimation.compute_hrf(bold_sig, para, temporal_mask, pjobs, bf=bf) '''

# so passing in orthogonalized bf (101, 3)

import os
import shutil
import tempfile
import numpy as np
from scipy        import stats
from scipy.sparse import lil_matrix
from joblib       import load, dump
from joblib       import Parallel, delayed
from rsHRF        import processing, sFIR
from rsHRF.rsHRF.processing import knee
from rsHRF.rsHRF.utils import hrf_estimation

def compute_hrf(bold_sig, para, temporal_mask, p_jobs, bf = None):
    para['temporal_mask'] = temporal_mask
    N, nvar = bold_sig.shape # N = 1200, nvar = 31
    folder = tempfile.mkdtemp()
    data_folder = os.path.join(folder, 'data') #random temp folder: /tmp/tmpbyc2xzwl/data
    dump(bold_sig, data_folder)
    data = load(data_folder, mmap_mode='r') # the same (1200, 31)
    results = Parallel(n_jobs=p_jobs)(delayed(estimate_hrf)(data, i, para,
                                  N, bf) for i in range(nvar)) # running estimate_hrf for each ROI in nvar (31) with the provided basis functions
    beta_hrf, event_bold = zip(*results)
    try:
        shutil.rmtree(folder)
    except:
        print("Failed to delete: " + folder)
    return np.array(beta_hrf).T, np.array(event_bold)


def estimate_hrf(bold_sig, i, para, N, bf = None):
    """
    Estimate HRF
    """
    dat = bold_sig[:, i] #(1200,) extract the tcs for just this ROI
    localK = para['localK'] #1
    if para['estimation'] == 'sFIR' or para['estimation'] == 'FIR':
        #Estimate HRF for the sFIR or FIR basis functions
        if np.count_nonzero(para['thr']) == 1:
            para['thr'] = np.array([para['thr'], np.inf])
        thr = para['thr'] #Thr is a vector for (s)FIR
        u = wgr_BOLD_event_vector(N, dat, thr, localK, para['temporal_mask'])
        u = u.toarray().flatten('C').ravel().nonzero()[0] # [] ?????
        beta_hrf, event_bold = sFIR.smooth_fir.wgr_FIR_estimation_HRF(u, dat, para, N)
        
        
    else:
        thr = [para['thr']] #Thr is a scalar for the basis functions #[1] 
        
        # DETECT BOLD EVENTS
        u0 = wgr_BOLD_event_vector(N, dat, thr, localK, para['temporal_mask']) # outputs a sparse matrix with shape (1, N) with 1s at event timepoints
        
        # PADDING WITH 0s TO CREATE MICROTIME RES GRID
        u = np.append(u0.toarray(), np.zeros((para['T'] - 1, N)), axis=0) # creates placeholder space (T - 1 rows of 0s) for the upsampled temporal grid
        # The HRF estimation requires a finer temporal grid to accurately model the hemodynamic response
        # para['T'] represents the "magnification factor of the temporal grid with respect to TR". It's used for temporal upsampling
        # so it goes from (1, 1200) to (3, 1200)

        # RESHAPING IN PREPARATION FOR CONVOLUTION
        u = np.reshape(u, (1, - 1), order='F')
        # basically flattens it all - reshaped to (1, 3600)
        
        beta_hrf = wgr_hrf_fit(dat, para, u, bf) # get the optimal weights for each basis function and the optimal time lag 
        
        u = u0.toarray()[0].nonzero()[0]
        
    return beta_hrf, u


'''
[0, 0, 1, 0, 0, 0, 0, 1, 0, ...] (shape 1×N)

[0, 0, 1, 0, 0, 0, 0, 1, 0, ...]  (original events)
[0, 0, 0, 0, 0, 0, 0, 0, 0, ...]  (zeros row 1)
[0, 0, 0, 0, 0, 0, 0, 0, 0, ...]  (zeros row 2)

[0, 0, 0, 0, 0, 0, ..., 1, 0, 0, 0, 0, 0, ...] (shape 1×(T*N))
'''

def wgr_BOLD_event_vector(N, matrix, thr, k, temporal_mask):
    """
    Detect BOLD event.
    event > thr & event < 3.1
    """
    #scipy.sparse lil_matrix - list of list matrix, a sparse matrix designed to efficiently work on matrices that are mostly zeros 
    # Only a small fraction of timepoints are identified as "events" (marked with 1s) - most BOLD signal is not 
    # using a sparse matrix saves memory
    data = lil_matrix((1, N))     # this makes data (1, 1200)
    matrix = matrix[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    if 0 in np.array(temporal_mask).shape:
        matrix = stats.zscore(matrix, ddof=1)
        for t in range(1 + k, N - k + 1): # from range (2, 1200)
        #if this point is greater than threshold ([0] for column is just bc it's a (N, 1) array)
            if matrix[t - 1, 0] > thr[0] and \
                    np.all(matrix[t - k - 1:t - 1, 0] < matrix[t - 1, 0]) and \
                    np.all(matrix[t - 1, 0] > matrix[t:t + k, 0]): 
                    # make sure the current point is greater than all k points behind it, and greater than all k points ahead of it
                    # np.all returns a boolean True or False
                data[0, t - 1] = 1 # if all conditions met, this current point is marked as a spontaneous neural 'event'
                # in this sparse binary matrix
    else:
        tmp = temporal_mask
        for i in range(len(temporal_mask)):
            if tmp[i] == 1:
                temporal_mask[i] = i
        datm = np.mean(matrix[temporal_mask])
        datstd = np.std(matrix[temporal_mask])
        if datstd == 0: datstd = 1
        matrix = (matrix - datm)/datstd
        for t in range(1 + k, N - k + 1):
            if tmp[t-1]:
                if matrix[t - 1, 0] > thr[0] and \
                        np.all(matrix[t - k - 1:t - 1, 0] < matrix[t - 1, 0]) and \
                        np.all(matrix[t - 1, 0] > matrix[t:t + k, 0]):
                    data[0, t - 1] = 1.
    return data

def wgr_hrf_fit(dat, xBF, u, bf):
    """
    @u    - BOLD event vector (microtime). 
    @nlag - time lag from neural event to BOLD event
    """
    lag = xBF['lag']
    AR_lag = xBF['AR_lag']
    nlag = len(lag)
    erm = np.zeros((1, nlag)) #create array to store errors
    beta = np.zeros((bf.shape[1] + 1, nlag)) #create array to store beta regression coefficients
    
    for i in range(nlag): # goes through each potential time lag in the 'lag' integer array
    
        u_lag = np.append(u[0, lag[i]:], np.zeros((1, lag[i]))).T # creates a new u event vector that is shifted back lag[i]i time bins 
            # slices the original event vector, starting at the position lag[i], then appends a vector of zeros with length lag[i] at the end
            # result: a new event vector where each "event" is now considered to have happened lag[i] time bins earlier than originally detected
            # shape (3600,)
            
         # Stores the resulting error and beta coefficients
        erm[0, i], beta[:, i] = wgr_glm_estimation(dat, u_lag, bf, xBF['T'], xBF['T0'], AR_lag) # Calls wgr_glm_estimation to fit the model with this lag
            # returns residual sum and beta coeffs 
            
    x, idx = knee.knee_pt(np.ravel(erm)) # Calls knee_pt to find the "knee point" in the error values that gives optimal lag
    if idx == nlag-1:
        idx = idx - 1 # Adjusts the index if necessary
    beta_hrf = beta[:, idx+1] # Extracts the beta coefficients for the optimal lag
    beta_hrf = np.append(beta_hrf, lag[idx+1]) # Appends the optimal lag value to the coefficients
    
    return beta_hrf

    # final output:
        # set of β coefficients (weights for the basis functions) that, when applied to this basis set, 
        # produce the best-fitting HRF for this particular brain region, at the optimal time lag value
        
        # resulting region-specific HRF that we construct is a weighted combination of the basis functions, 
        # with weights determined by the β values of the optimal lag

def wgr_glm_estimation(dat, u, bf, T, T0, AR_lag):
    """
    @u - BOLD event vector (microtime).
    """
    nscans = dat.shape[0] #N , 1200
    x = wgr_onset_design(u, bf, T, T0, nscans) # creates design matrix
    X = np.append(x, np.ones((nscans, 1)), axis=1) # Appends a column of ones to the design matrix (this represents the intercept term)
    
    res_sum, Beta = sFIR.smooth_fir.wgr_glsco(X, dat, AR_lag=AR_lag)
    # call wgr_glsco to solve the GLM using generalized least squares (GLS) - (better than OLS bc it can account for autocorrelation)?
    # goal: to solve the GLM to find the weights (β) for each basis function that best explain the observed data
    # each regressor is the predicted BOLD response acc. to each basis function - find optimal beta coeffs for each regressor that results in lowest res_sum
    
    return np.real(res_sum), Beta # returns residual sum: sum of the squared differences between the observed data and the model's prediction
    # a lower residual sum indicates a better fit


def wgr_onset_design(u, bf, T, T0, nscans): 
    """
    @u - BOLD event vector (microtime). # input time-shifted event vector (3600,) at upsampled microtime resolution
    @bf - basis set matrix # (101, 3) still
    @T - microtime resolution (number of time bins per scan)
    @T0 - microtime onset (reference time bin, see slice timing)
    @nscans - number of actual fMRI scans/timepoints (ie 1200)
    
    creates the design matrix for GLM analysis
    """
    ind = np.arange(0, max(u.shape)) # Creates an array of indices for the full length of the event vector
    X = np.empty((0, len(ind))) # Initializes an empty array X that will become our design matrix
    
    for p in range(bf.shape[1]): # for each basis function (**remember each basis function represents a diff component of the HRF - we orthogonalized)
        x = np.convolve(u, bf[:, p]) # Convolve the event vector u with the basis function - create expected BOLD response based on this basis function
        x = x[ind] # Extract relevant values from the convolution result
        X = np.append(X, [x], axis=0) # Add this as a row to the design matrix X
        
        
    X = X.T # Transpose X so that each column represents a regressor (basis function convolved with events)
    """
    Resample regressors at acquisition times
    """
    
    # downsample back from microtime to actual TR (0.24s back to 0.72s timestep)
    X = X[(np.arange(0, nscans) * T) + (T0 - 1), :]
    return X

    # so final output is X design matrix: 
        # with the predicted BOLD signal according to each independent basis function
        # back to BOLD TR temporal res (1200 scans at 0.72s resolution)
        # one column for each basis function
        # one row for each BOLD TR timepoint

#%% 
# let's say i=1

i=1
dat = bold_sig[:, i]
localK = para['localK']
thr = [para['thr']]

u0 = wgr_BOLD_event_vector(N, dat, thr, localK, para['temporal_mask']) # a sparse lil matrix object that is (1, 1200)
u = np.append(u0.toarray(), np.zeros((para['T'] - 1, N)), axis=0) # (3, 1200) # zero padding, added 2 rows of zeros # create upsampled microtime temporal grid
u = np.reshape(u, (1, - 1), order='F') # now reshaped to (1, 3600) - flatten to prepare for convolution

        
# previously in evaluate() lag was set up like this:        
para['lag'] = np.arange(np.fix(para['min_onset_search'] / para['dt']),    
                    np.fix(para['max_onset_search'] / para['dt']) + 1,
                    dtype='int') # an array of (22,) from [12 13 14 .... 32 33] 

'''
# min_onset_search and max set the search window
# para['dt'] is the time bin size for basis functions in seconds (typically TR/T, where T is the upsampling factor)
# np.fix() truncates the decimal part 
# so 'lag' is an integer array of the different time lags to test

# each int represents the number of TIME BINS (each bin represents dt=0.24s of true time) to shift the event vector backwards - to estimate actual neural event onset
# ie. wgr_BOLD_event_vector detects the PEAKS in BOLD signal - creates event vector 
# then test each lag in the array, fit a GLM, pick the one with the least error (aka best fit) - so we are estimating optimal lag and neural event onset simultaneously
# so if best lag turns out to be 12 for this ROI that means T2P is generally 12*dt (0.24s) = 2.88s

NOTE: autocorrelation lag is a statistical correction: BOLD signals are temporally correlated - violates assumptions of GLM
so this param determines how many previous timepoints to consider when modeling this autocorrelation

'''


lag = para['lag'] # array of (22,) 
AR_lag = para['AR_lag'] # autocorrelation lag, set to 1
nlag = len(lag) # 22 
erm = np.zeros((1, nlag)) # (1, 22)
beta = np.zeros((bf.shape[1] + 1, nlag)) # (4, 22)


'''
so now we are:
    - assessing one brain region at a time
    - testing one specific time lag at a time (# time bins the BOLD peaks should be shifted back) 
    - We have a set of hypothesized neural event times (the time-shifted event vector)
    - We need to see how well these events, when convolved with the HRF basis functions, explain the observed BOLD signal                         
'''
for i in range(nlag):
    u_lag = np.append(u[0, lag[i]:], np.zeros((1, lag[i]))).T # u_lag is now (3600,) - shape didn't change
    
    #erm[0, i], beta[:, i] = \
        #wgr_glm_estimation(dat, u_lag, bf, xBF['T'], xBF['T0'], AR_lag)
        
nscans = dat.shape[0] #N , 1200
ind = np.arange(0, max(u_lag.shape)) # (3600) [0 1 2 ... 3598 3599 ]
X = np.empty((0, len(ind))) # (0, 3600) [0]

for p in range(bf.shape[1]): # for each basis function (3 total here)
    x = np.convolve(u_lag, bf[:, p]) # convolve this time-shifted event vector with this basis function #(3700,)
    x = x[ind] # extract out only the full event vector length we need (back to (3600,))
    X = np.append(X, [x], axis=0) # (1, 3600)
    
# by the end: X is (3, 3600) 
X = X.T #(3600, 3)
X = X[(np.arange(0, nscans) * para['T']) + (para['T0'] - 1), :] # downsampled back to (1200, 3)

# GLM solution
X = np.append(X, np.ones((nscans, 1)), axis=1) # Appends a column of ones to the design matrix (this represents the intercept term) # (1200, 4)

res_sum, Beta = sFIR.smooth_fir.wgr_glsco(X, dat, AR_lag=AR_lag)
# ERROR: ValueError: shape mismatch: value array of shape (4,1) could not be broadcast to indexing result of shape (4,)
# File /mnt/storage/HRF_deconvolution/rsHRF/rsHRF/sFIR/smooth_fir.py:27 in wgr_regress
  #  b[perm] = linalg.solve(R,np.matmul(Q.T,y))
  
# wgr_glsco and wgr_regress expect dat to be a 1D array of shape (1200,)
    
# just try flattening it: change from (1200, 1) to (1200,) ***
dat_reshaped = dat.flatten()
res_sum, Beta = sFIR.smooth_fir.wgr_glsco(X, dat_reshaped, AR_lag=AR_lag) 
# residual sum is 0.836831??? array of 1
# Beta is (4,) : array([10.60939445,  8.3531353 , -4.73054718, -0.01634912])


#%% 
# what if I run it on all - actually find knee point and do it for all regions

from scipy import signal, stats
        
from rsHRF.utils import hrf_estimation 
para['temporal_mask'] = []
beta_hrf, event_bold = hrf_estimation.compute_hrf(bold_sig, para, para['temporal_mask'], para['pjobs'], bf=bf)
# beta_hrf is (5, 31): beta coeffs for the 3 basis functions, then the intercept column (the baseline offset), 
# then the optimal time lag (knee point) identified for that ROI

''' so betas summarized 
for each ROI, there are 5 params that characterize the HRF:
    1. Weight for canonical HRF: How much the standard hemodynamic response function contributes
    2. Weight for temporal derivative: How much to adjust the timing (peak latency) of the HRF
    3. Weight for dispersion derivative: How much to adjust the width/duration of the HRF
    4. Intercept term: The baseline offset (doesn't affect shape itself)
    5. Optimal time lag: The number of time bins of delay between neural events and BOLD response that gave the best model fit (appended in hrf_fit)

'''

hrfa = np.dot(bf, beta_hrf[np.arange(0, bf.shape[1]), :]) # (101, 31) # creates the region-specific HRF as a combination of the basis functions, weighted by the corresponding betas
# Only take the first 3 values (betas for the 3 basis functions), Multiply them by their respective basis functions, Sum these weighted basis functions        
# matrix multiplication between: bf (shape (101, 3)) and extracted just the first 3 betas (for the 3 bfs) - (shape (3, 31))
hrf = hrfa.T # (101, 31) - now each row is the HRF of a region (101 timepoints, each is 0.24s)

plt.plot(hrfa)
plt.show()
plt.hrfa # this looks insanely better - so much variation! 

upsample = lambda x : signal.resample_poly(x[::-1], 6000, hrf.shape[1]) # var.shape[0]

# can't run this by itself because it needs var but this is dependent on the monitor object (stock step size)
final_return = np.apply_along_axis(upsample, 1, hrf) #(31, 6000)
plt.plot(final_return.T[::-1])

## TRYING RESAMPLE METHOD INSTEAD

resample = lambda x : signal.resample(x[::-1], 6000)

resampled_return = np.apply_along_axis(resample, 1, hrf) # (31, 6000)
plt.plot(resampled_return.T[::-1])

## SO WHAT WAS THE PROBLEM WITH RESAMPLE_POLY

upsample_fixed = lambda x : signal.resample_poly(x[::-1], 6000, hrf.shape[1]) # var.shape[0]
upsample_fixed_return = np.apply_along_axis(upsample_fixed, 1, hrf) #(31, 6000)

# unclear 
# up: 6000
# down: 101

# resampling ratio of 6000/101 ≈ 59.4 
# this is right - each point becomes 60 = 100 * 60 = 6000

#%% 

# so go back to following Amogh Johri's notebook:
    
''' 
technically each subclass of Equation's evaluate() method is called on evaluate(self, var) and 'var' represents a distance or effective distance per node in a sim

# From the BOLD monitor's compute_hrf method
self._stock_sample_rate = 2.0**-2  # 0.25 ms⁻¹
magic_number = self.hrf_length  # 24000 
required_history_length = self._stock_sample_rate * magic_number  # 0.25 * 24000 = 6000
self._stock_steps = numpy.ceil(required_history_length).astype(int)  # 6000

self._stock_sample_rate is fixed at 0.25 ms⁻¹
self._stock_steps - the number of time points needed to represent the full HRF at the monitor's internal sampling rate for convolution operations

the upsample line is:
resampling from 101 points to 6000 points, while also reversing the HRF in time


'''

file_path = '/mnt/storage/neuronumba/demos/2_MIST_parcellations/031-MIST/Data_Raw_031-MIST/sub-101309/rfMRI_REST1_LR_BOLD.csv'
roiTimeSeries = np.loadtxt(file_path)
print("Shape of the BOLD time-series: ", roiTimeSeries.shape)
# Shape of the BOLD time-series:  (1200, 31)

bold_rsHRF_roiTS = monitors.Bold(hrf_kernel = equations.RestingStateHRF(roiTS = roiTimeSeries))

temporal_mask = []                                   
bold_rsHRF_roiTS.dt = 0.72 #ms, because this is coming out of temporal average monitor
bold_rsHRF_roiTS.period = 720
bold_rsHRF_roiTS.hrf_length = 24000
bold_rsHRF_roiTS.configure()
bold_rsHRF_roiTS

bold_rsHRF_roiTS.compute_hrf()
hrf_roiTS  = bold_rsHRF_roiTS.hemodynamic_response_function
print("Shape of the hrf is: ", hrf_roiTS.shape) #(31, 6000)


# plot the kernel
plt.plot(bold_rsHRF_roiTS._stock_time, hrf_roiTS[0].T[::-1])
plt.ylabel('hrf')
plt.xlabel('time [sec]')
plt.plot(bold_rsHRF_roiTS._stock_time[hrf_roiTS[0][::-1].argmax()], hrf_roiTS[0][::-1].max(), 'ko')

# plot every region

plt.close()
plt.plot(bold_rsHRF_roiTS._stock_time, hrf_roiTS.T[::-1])


#%% trying resample method







#%%

from scipy.io import loadmat
hrfs = loadmat(filename)
bfs_mat = hrfs['bf']
hrfa_mat = hrfs['hrfa']

#%% using regular TVB library

from tvb.simulator import simulator, models, coupling, monitors

# sim = simulator.Simulator(
#     model=models.ReducedWongWangExcInh(),  # Or any neural model
#     connectivity=connectivity,  # Your brain connectivity
#     coupling=coupling.Linear(),  # Coupling function
#     integrator=integrators.HeunDeterministic(dt=0.09),  # Integration scheme
#     monitors=[bold_rsHRF_roiTS],  # Your BOLD monitor with region-specific HRFs
#     simulation_length=60000  # 60 seconds
# )

# sim.configure()
# results = sim.run()
# bold_time, bold_data = results[0]  # Extract BOLD time series


#%% using neuronumba

# raw neural tcs (temporal monitor, 0.72ms timesteps)
import numpy as np
import h5py
filename = '/mnt/storage/neuronumba/demos/2_MIST_parcellations/031-MIST/Data_Produced_031-MIST/Deco2014_FC/timeseries/temp_avg_neural_tcs_1.80.mat'

with h5py.File(filename, 'r') as f:
    print(list(f.keys()))
    
    sub_101309_REST1_LR_sim_neural = f['subject_0'][:] # is (1227779, 31) # 1.2 million timepoints at 0.72ms resolution = 884000ms true time simulated --> 20s warmup

np.savetxt('/mnt/storage/HRF_deconvolution/region-specific-hrfs-microtime.txt', hrfa)

from neuronumba.bold import rsHRF
bold_model = BoldRegionSpecificHRF(
    tr=720.0, 
    hrf_filename="/mnt/storage/HRF_deconvolution/region-specific-hrfs-microtime.txt"
).configure()

# Compute BOLD signal
bold_signal = bold_model.compute_bold(neural_signal, 0.09)  # dt = 0.09 ms