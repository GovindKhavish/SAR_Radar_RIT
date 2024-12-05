#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import skimage
import Spectogram_Functions
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
from scipy.signal import windows
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt
from sklearn.cluster import DBSCAN
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path, PurePath
#-----------------------------------------------------------------------------------------
# Define the subdirectory path
_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

# Check if the subdirectory exists
if _simraddir.exists():
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder

#-----------------------------------------------------------------------------------------
# Mipur VH
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'
inputfile = filepath + filename

# Level 0 File
l0file = sentinel1decoder.Level0File(inputfile)

# Metadata and Burst Information
sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

# Select the Burst
selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

headline = f'Sentinel-1 (burst {selected_burst}): '

# Raw I/Q Sensor Data
radar_data = l0file.get_burst_data(selected_burst)

# ------------------ Spectrogram Plot with Local Adaptive Thresholding -------------------
def adaptive_threshold_local(data, threshold_factor):
    mean_val = np.mean(data)
    std_val = np.std(data)
    threshold = mean_val + threshold_factor * std_val

    thresholded_data = np.where(data > threshold, data, 0)
    
    return threshold,thresholded_data


def adaptive_threshold_sliding_window(data, window_size, threshold_factor):
    mean_filter = uniform_filter(data, size=window_size)
    threshold = mean_filter + threshold_factor
    thresholded_data = np.where(data > threshold, data, 0)
    
    return threshold, thresholded_data

import numpy as np
from scipy.ndimage import gaussian_filter

def quick_preprocessing(data, sigma=1, percentile=90):
    """
    Lightweight preprocessing pipeline for spectrograms.
    
    Parameters:
    - data: 2D numpy array of spectrogram data
    - noise_floor: Minimum intensity value (dB) to suppress low noise
    - sigma: Gaussian smoothing factor
    - percentile: Percentile for adaptive thresholding
    """
    # Step 1: Suppress noise below a floor
    noise_floor = -50#np.mean(aa)
    print(noise_floor)
    data[data < 10 ** (noise_floor / 10)] = 10 ** (noise_floor / 10)

    # Step 2: Apply logarithmic scaling
    log_data = 10 * np.log10(data)

    # Step 3: Smooth the spectrogram with a Gaussian filter
    smoothed_data = gaussian_filter(log_data, sigma=sigma)

    # Step 4: Perform percentile-based adaptive thresholding
    threshold = np.percentile(smoothed_data, percentile)
    thresholded_data = np.where(smoothed_data > threshold, smoothed_data, 0)
    
    return threshold,thresholded_data


def spectral_subtraction(data, noise_estimation_factor=0.5):
    """
    Performs spectral subtraction to remove background noise.
    
    Parameters:
    - data: 2D numpy array of spectrogram data.
    - noise_estimation_factor: Factor for estimating noise (0.5 typically).
    
    Returns:
    - enhanced_data: The spectrogram with noise reduced.
    """
    noise_estimation = np.min(data, axis=0)  # Estimate background noise (min value per time slice)
    enhanced_data = data - noise_estimation_factor * noise_estimation
    enhanced_data = np.clip(enhanced_data, 0, None)  # Remove any negative values (artifact)
    return enhanced_data

def adaptive_local_thresholding(data, window_size=5, threshold_factor=2):
    """
    Applies local adaptive thresholding based on mean and std in local windows.
    
    Parameters:
    - data: 2D numpy array of spectrogram data.
    - window_size: Size of the local window for computing mean and std.
    - threshold_factor: Factor to scale the threshold based on local std.
    
    Returns:
    - thresholded_data: Data after applying adaptive thresholding.
    """
    local_mean = median_filter(data, size=window_size)
    local_std = gaussian_filter(data, sigma=window_size)
    
    threshold = local_mean + threshold_factor * local_std
    thresholded_data = np.where(data > threshold, data, 0)
    return thresholded_data

def enhanced_preprocessing(data, noise_estimation_factor=0.5, window_size=5, threshold_factor=2):
    """
    Enhanced preprocessing pipeline for spectrograms using spectral subtraction
    and adaptive thresholding.
    
    Parameters:
    - data: 2D numpy array of spectrogram data.
    - noise_estimation_factor: Factor for estimating noise.
    - window_size: Size of the local window for thresholding.
    - threshold_factor: Factor for adaptive thresholding.
    
    Returns:
    - processed_data: Preprocessed spectrogram ready for DBSCAN.
    """
    # Step 1: Spectral Subtraction
    enhanced_data = spectral_subtraction(data, noise_estimation_factor=noise_estimation_factor)
    
    # Step 2: Apply adaptive thresholding
    thresholded_data = adaptive_local_thresholding(enhanced_data, window_size=window_size, threshold_factor=threshold_factor)
    
    # Optional: Clean up using morphological closing to remove small isolated points
    cleaned_data = closing(thresholded_data, disk(3))  # 3 is the disk size for morphological operation
    
    return cleaned_data


# ------------------ Plotting -------------------
idx_n = 150
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

# fig = plt.figure(10, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)
# ax.plot(np.abs(radar_section), label=f'abs{idx_n}')
# ax.plot(np.real(radar_section), label=f'Re{idx_n}')
# ax.plot(np.imag(radar_section), label=f'Im{idx_n}')
# ax.legend()
# ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold')
# ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold')
# ax.set_ylabel('|Amplitude|', fontweight='bold')
# plt.tight_layout()
# plt.pause(0.1)

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'
aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# Apply adaptive threshold
#threshold, aa_db_filtered  = adaptive_threshold_sliding_window(aa, 7,2)
threshold, aa_db_filtered  = adaptive_threshold_local(aa,2)
#threshold, aa_db_filtered  = quick_preprocessing(aa)
#aa_db_filtered  = enhanced_preprocessing(aa)

# Plot the filtered spectrogram
fig = plt.figure(12, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
dd = ax.imshow(10 * np.log10(aa_db_filtered), aspect='auto', origin='lower', cmap='Greys', extent=[0, aa.shape[1], bb[0], bb[-1]])
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Filtered Spectrogram (Threshold:  dB)', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

thresholded_aa_flat = aa_db_filtered .flatten()

# 2D (time, frequency)
time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points

# Frequency bins
frequency_indices = bb[time_freq_data[:, 0]]

# DBSCAN
dbscan = DBSCAN(eps=2, min_samples=10)
clusters = dbscan.fit_predict(time_freq_data)

# Plot threshold
fig_thresh = plt.figure(13, figsize=(6, 6), clear=True)
ax_thresh = fig_thresh.add_subplot(111)
aa_db_filtered = 10 * np.log10(aa_db_filtered  + 1e-10) 
dd = ax_thresh.imshow(aa_db_filtered, aspect='auto', origin='lower', cmap='Greys')

# Plot clusters 
for i, cluster in enumerate(np.unique(clusters)):
    if cluster != -1:  # -1 is noise
        cluster_points = time_freq_data[clusters == cluster]
        ax_thresh.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i}', s=2)

cbar = plt.colorbar(dd, ax=ax_thresh)
cbar.set_label('Intensity [dB]')
ax_thresh.set_xlabel('Time [us]', fontweight='bold')
ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
ax_thresh.set_title(f'Filtered Spectrogram with DBSCAN (Threshold: dB)', fontweight='bold')
ax_thresh.legend()
plt.tight_layout()
#{round(10 * np.log10(threshold), 2)}


# Number of clusters (noise is -1)
num_clusters = len(np.unique(clusters)) - 1 
print(f"Number of clusters (excluding noise): {num_clusters}")

# ------------------ Start and End Time for Each Cluster -------------------
cluster_time_indices = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Noise

        # Time-freq points for cluster
        cluster_points = time_freq_data[clusters == cluster_id]
        # Time indices (2nd column)
        time_indices = cluster_points[:, 1]  # Time axis (us)
        
        # Start and End time
        start_time_index = np.min(time_indices)
        end_time_index = np.max(time_indices)

        cluster_time_indices[cluster_id] = (start_time_index, end_time_index)

for cluster_id, (start, end) in cluster_time_indices.items():
    print(f"Cluster {cluster_id}: Start Time Index = {start}, End Time Index = {end}")

# Extract Cluster Parameters
cluster_params = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude noise 
        cluster_points = time_freq_data[clusters == cluster_id]
        frequency_indices = bb[cluster_points[:, 0]]  # Use the correct frequency bins (bb)
        
        # 2nd column of the time_freq_data
        time_indices = cluster_points[:, 1]  # us
        

        bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
        center_frequency = np.mean(frequency_indices)
        time_span = np.max(time_indices) - np.min(time_indices)  # us
        if time_span != 0:
            chirp_rate = bandwidth / time_span  # MHz per us
        else:
            chirp_rate = 0 
        
        cluster_params[cluster_id] = {
            'bandwidth': bandwidth,
            'center_frequency': center_frequency,
            'chirp_rate': chirp_rate,
            'start_time_index': np.min(time_indices),
            'end_time_index': np.max(time_indices)
        }

for cluster_id, params in cluster_params.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Bandwidth: {params['bandwidth']} MHz")
    print(f"  Center Frequency: {params['center_frequency']} MHz")
    print(f"  Chirp Rate: {params['chirp_rate']} MHz/us")
    print(f"  Start Time Index: {params['start_time_index']}")
    print(f"  End Time Index: {params['end_time_index']}")
    print('---------------------------')

# # Numpy Array conversition
# cluster_params_array = np.array([[cluster_id, params['bandwidth'], params['center_frequency'], params['chirp_rate'], params['start_time_index'], params['end_time_index']]
#                                  for cluster_id, params in cluster_params.items()])

NFFT = 256
noverlap = 200
sampling_rate = fs
time_step = (NFFT - noverlap) / sampling_rate  # seconds

# Convert to I/Q data
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

mapped_cluster_indices = {}
for cluster_id, params in cluster_params.items():
    start_time_idx = params['start_time_index']
    end_time_idx = params['end_time_index']
    iq_start_idx = spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
    iq_end_idx = spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
    mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

for idx in range(len(radar_section)):
    for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
        if iq_start_idx <= idx <= iq_end_idx:
            isolated_radar_data[idx] = radar_section[idx]
            break 

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.real(radar_section), label='Real Part', alpha=0.7, linestyle='-')
plt.plot(np.imag(radar_section), label='Imaginary Part', alpha=0.7, linestyle='-')
plt.title('Full Radar I/Q Data')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.real(isolated_radar_data), label='Real Part', alpha=0.7, linestyle='-')
plt.plot(np.imag(isolated_radar_data), label='Imaginary Part', alpha=0.7, linestyle='-')
plt.title('Isolated Radar I/Q Data (Zero Outside Clusters)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


