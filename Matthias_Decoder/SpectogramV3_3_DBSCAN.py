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
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
from scipy.signal import windows
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
import Spectogram_Functions
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
def adaptive_threshold_local(spectrogram_data, threshold_factor):
    mean_val = np.mean(spectrogram_data)
    std_val = np.std(spectrogram_data)
    threshold = mean_val + threshold_factor * std_val

    thresholded_data = np.where(spectrogram_data > threshold, spectrogram_data, 0)
    
    return threshold,thresholded_data

# ------------------ Plotting -------------------
idx_n = 150
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

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
threshold, aa_db_filtered  = adaptive_threshold_local(aa, 2)

# Plot the filtered spectrogram
fig = plt.figure(12, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
dd = ax.imshow(10 * np.log10(aa_db_filtered), aspect='auto', origin='lower', cmap='Greys', extent=[0, aa.shape[1], bb[0], bb[-1]])
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Filtered Spectrogram (Threshold: {round(10 * np.log10(threshold), 2)} dB)', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# DBSCAN 
thresholded_aa_flat = aa_db_filtered .flatten()

# 2D array for clustering (time, frequency)
time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points

# Map the frequency bins (bb) to the clustering logic
frequency_indices = bb[time_freq_data[:, 0]]

# DBSCAN clustering (same as before, but with proper frequency reference)
dbscan = DBSCAN(eps=2, min_samples=12)
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
ax_thresh.set_title(f'Filtered Spectrogram with DBSCAN (Threshold: {round(10 * np.log10(threshold), 2)} dB)', fontweight='bold')
ax_thresh.legend()
plt.tight_layout()

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
    if cluster_id != -1:  # Exclude noise points
        # Get the time-frequency points corresponding to the current cluster
        cluster_points = time_freq_data[clusters == cluster_id]
        
        # Get the frequency indices (now mapped to `bb`)
        frequency_indices = bb[cluster_points[:, 0]]  # Use the correct frequency bins (bb)
        
        # Get the time indices (the 2nd column of the time_freq_data)
        time_indices = cluster_points[:, 1]  # Column 1 is the time axis (in microseconds)
        
        # Bandwidth: Difference between the maximum and minimum frequency values
        bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
        
        # Center frequency: Average of the frequency values in the cluster
        center_frequency = np.mean(frequency_indices)
        
        # Chirp rate: Difference in frequency divided by the time span (approximated)
        time_span = np.max(time_indices) - np.min(time_indices)  # Time span in microseconds
        if time_span != 0:
            chirp_rate = bandwidth / time_span  # MHz per microsecond
        else:
            chirp_rate = 0  # In case the time span is zero (unlikely, but for safety)
        
        # Store the parameters for each cluster
        cluster_params[cluster_id] = {
            'bandwidth': bandwidth,
            'center_frequency': center_frequency,
            'chirp_rate': chirp_rate,
            'start_time_index': np.min(time_indices),
            'end_time_index': np.max(time_indices)
        }

# Print out the parameters for each cluster
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

# Define the spectrogram parameters
NFFT = 256
noverlap = 200
sampling_rate = fs  # Sampling rate of the I/Q data

# Calculate time step in samples
time_step = (NFFT - noverlap) / sampling_rate  # Time per spectrogram bin (in seconds)

# Convert spectrogram time indices to I/Q data indices
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

# Map cluster time indices to original I/Q data indices
mapped_cluster_indices = {}
for cluster_id, params in cluster_params.items():
    start_time_idx = params['start_time_index']
    end_time_idx = params['end_time_index']
    iq_start_idx = spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
    iq_end_idx = spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
    mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

# Initialize isolated data with zeros
isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

# Iterate over the I/Q data
for idx in range(len(radar_section)):
    for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
        if iq_start_idx <= idx <= iq_end_idx:
            isolated_radar_data[idx] = radar_section[idx]
            break  # Stop checking once matched

# Plot the full radar data
plt.figure(figsize=(12, 6))

# Full radar data plot
plt.subplot(2, 1, 1)
plt.plot(np.real(radar_section), label='Real Part', alpha=0.7, linestyle='-')
plt.plot(np.imag(radar_section), label='Imaginary Part', alpha=0.7, linestyle='-')
plt.title('Full Radar I/Q Data')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Isolated radar data plot
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


