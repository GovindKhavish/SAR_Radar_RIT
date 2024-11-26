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
import pprint 
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

# Mipur VH Filepath
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

# ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
def adaptive_threshold_local(spectrogram_data, threshold_factor):
    mean_val = np.mean(spectrogram_data)
    std_val = np.std(spectrogram_data)
    threshold = mean_val + threshold_factor * std_val

    thresholded_data = np.where(spectrogram_data > threshold, spectrogram_data, 0)
    
    return threshold, thresholded_data

# Convert spectrogram time indices to I/Q data indices
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

# Global pulse counter
global_pulse_number = 1

# Define the range of rangelines you want to process
start_idx = 0  # Start rangeline index
end_idx = 150   # End rangeline index
fs = 46918402.800000004  # Sampling rate in Hz (samples per second)

# Global dictionary to hold cluster parameters for all rangelines
global_cluster_params = {}

# Loop through the specified rangeline indices
for idx_n in range(start_idx, end_idx + 1):
    # Extract the radar data for the current rangeline
    radar_section = radar_data[idx_n, :]

    # Calculate slow time offset for this rangeline
    slow_time_offset = idx_n / fs  # Time offset in seconds for the current rangeline

    # ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
    fig = plt.figure(11, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(111)
    scale = 'dB'
    aa, bb, cc, dd = ax.specgram(
        radar_data[idx_n, :], 
        NFFT=256, 
        Fs=fs / 1e6, 
        detrend=None, 
        window=np.hanning(256), 
        scale=scale, 
        noverlap=200, 
        cmap='Greys'
    )

    # Apply adaptive threshold
    threshold, aa_db_filtered = adaptive_threshold_local(aa, 2)

    # ------------------ DBSCAN Clustering -------------------
    thresholded_aa_flat = aa_db_filtered.flatten()

    # 2D array for clustering (time, frequency)
    time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points

    # Map the frequency bins (bb) to the clustering logic
    frequency_indices = bb[time_freq_data[:, 0]]

    # DBSCAN clustering
    dbscan = DBSCAN(eps=2, min_samples=10)
    clusters = dbscan.fit_predict(time_freq_data)

    # Number of clusters (noise is -1)
    num_clusters = len(np.unique(clusters)) - 1
    print(f"Number of clusters for rangeline {idx_n}: {num_clusters}")

    # ------------------ Skip Feature Extraction if More Than 2 Clusters -------------------
    if num_clusters > 2:
        print(f"Skipping feature extraction for rangeline {idx_n} due to more than 2 clusters.")
        continue

    # ------------------ Assign Global Pulse Numbers and Adjusted Times -------------------
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Skip noise
            cluster_points = time_freq_data[clusters == cluster_id]
            frequency_indices = bb[cluster_points[:, 0]]
            time_indices = cluster_points[:, 1]
            bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
            center_frequency = np.mean(frequency_indices)
            time_span = np.max(time_indices) - np.min(time_indices)
            chirp_rate = bandwidth / time_span if time_span != 0 else 0

            # Compute adjusted start and end times
            start_time = np.min(time_indices) / fs  # Start time in seconds
            end_time = np.max(time_indices) / fs    # End time in seconds
            adjusted_start_time = start_time + slow_time_offset
            adjusted_end_time = end_time + slow_time_offset

            # Compute pulse duration
            pulse_duration = adjusted_end_time - adjusted_start_time

            # Create a unique key combining rangeline index and cluster ID
            unique_key = (idx_n, cluster_id)  # Use both rangeline index and cluster ID as a tuple

            # Assign cluster parameters along with the global pulse number
            if unique_key not in global_cluster_params:
                global_cluster_params[unique_key] = []

            # Store parameters for each cluster, ensuring we keep all rangeline data
            global_cluster_params[unique_key].append({
                'pulse_number': global_pulse_number,  # Use global pulse number
                'bandwidth': bandwidth,
                'center_frequency': center_frequency,
                'chirp_rate': chirp_rate,
                'start_time_index': np.min(time_indices),
                'end_time_index': np.max(time_indices),
                'adjusted_start_time': adjusted_start_time,  # Adjusted start time
                'adjusted_end_time': adjusted_end_time,      # Adjusted end time
                'pulse_duration': pulse_duration             # Pulse duration
            })
            global_pulse_number += 1  # Increment global pulse number for the next cluster

    # ------------------ Process I/Q Data -------------------
    # Define the spectrogram parameters
    NFFT = 256
    noverlap = 200
    sampling_rate = fs  # Sampling rate of the I/Q data

    # Calculate time step in samples
    time_step = (NFFT - noverlap) / sampling_rate  # Time per spectrogram bin (in seconds)

    cluster_time_indices = {}
    for (rangeline_idx, cluster_id), params_list in global_cluster_params.items():
        for params in params_list:
            start_time_idx = params['start_time_index']
            end_time_idx = params['end_time_index']
            iq_start_idx = spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
            iq_end_idx = spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
            cluster_time_indices[(rangeline_idx, cluster_id)] = (iq_start_idx, iq_end_idx)

    # Initialize isolated data with zeros
    isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

    # Iterate over the I/Q data
    for idx in range(len(radar_section)):
        for (rangeline_idx, cluster_id), (iq_start_idx, iq_end_idx) in cluster_time_indices.items():
            if iq_start_idx <= idx <= iq_end_idx:
                isolated_radar_data[idx] = radar_section[idx]
                break  # Stop checking once matched


# print("Cluster Parameters for All Pulses:")
# pprint.pprint(global_cluster_params)



# Initialize lists to store pulse number, bandwidth, and pulse duration
pulse_numbers = []
bandwidths = []
durations = []

# Extract values from global_cluster_params
for unique_key, params_list in global_cluster_params.items():
    for params in params_list:
        pulse_numbers.append(params['pulse_number'])
        bandwidths.append(params['bandwidth'])
        durations.append(params['pulse_duration'])

# Ensure data is sorted by pulse number for better visualization
sorted_indices = np.argsort(pulse_numbers)
pulse_numbers = np.array(pulse_numbers)[sorted_indices]
bandwidths = np.array(bandwidths)[sorted_indices]
durations = np.array(durations)[sorted_indices]

# Plot Pulse Number vs Bandwidth
plt.figure(figsize=(10, 5))
plt.plot(pulse_numbers, bandwidths, linestyle='-', color='b', label="Bandwidth")
plt.title("Pulse Number vs Bandwidth")
plt.xlabel("Pulse Number")
plt.ylabel("Bandwidth (MHz)")
plt.grid(True)
plt.legend()

# Plot Pulse Number vs Duration
plt.figure(figsize=(10, 5))
plt.plot(pulse_numbers, durations, linestyle='-', color='g', label="Duration")
plt.title("Pulse Number vs Pulse Duration")
plt.xlabel("Pulse Number")
plt.ylabel("Pulse Duration (us)")
plt.grid(True)
plt.legend()
plt.show()


# # Dictionary to store clusters and their specific information for each rangeline
# clusters_info = {}

# # Loop through the rangeline indices
# for idx_n in range(start_idx, end_idx + 1):
#     # Extract the radar data for the current rangeline
#     radar_section = radar_data[idx_n, :]

#     # Create an entry for this rangeline in the clusters_info dictionary
#     clusters_info[idx_n] = {
#         'clusters': {},  # To store the clusters for the current rangeline
#         'radar_data': radar_section  # Store the radar data as well (optional)
#     }

#     # ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
#     fig = plt.figure(11, figsize=(6, 6), clear=True)
#     ax = fig.add_subplot(111)
#     scale = 'dB'
#     aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale, noverlap=200, cmap='Greys')

#     # Apply adaptive threshold
#     threshold, aa_db_filtered = adaptive_threshold_local(aa, 2)

#     # DBSCAN Clustering
#     thresholded_aa_flat = aa_db_filtered.flatten()

#     # 2D array for clustering (time, frequency)
#     time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points

#     # Map the frequency bins (bb) to the clustering logic
#     frequency_indices = bb[time_freq_data[:, 0]]

#     # DBSCAN clustering
#     dbscan = DBSCAN(eps=3, min_samples=10)
#     clusters = dbscan.fit_predict(time_freq_data)

#     # Extract and Process I/Q Data for Clusters
#     cluster_params = {}
#     for cluster_id in np.unique(clusters):
#         if cluster_id != -1:  # Exclude noise
#             cluster_points = time_freq_data[clusters == cluster_id]
#             frequency_indices = bb[cluster_points[:, 0]]
#             time_indices = cluster_points[:, 1]
#             bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
#             center_frequency = np.mean(frequency_indices)
#             time_span = np.max(time_indices) - np.min(time_indices)
#             chirp_rate = bandwidth / time_span if time_span != 0 else 0
#             cluster_params[cluster_id] = {
#                 'bandwidth': bandwidth,
#                 'center_frequency': center_frequency,
#                 'chirp_rate': chirp_rate,
#                 'start_time_index': np.min(time_indices),
#                 'end_time_index': np.max(time_indices)
#             }

#     # Store the cluster parameters for the current rangeline
#     clusters_info[idx_n]['clusters'] = cluster_params

#     # Optionally, print out the cluster parameters for each rangeline
#     for cluster_id, params in cluster_params.items():
#         print(f"Cluster {cluster_id} for rangeline {idx_n}:")
#         print(f"  Bandwidth: {params['bandwidth']} MHz")
#         print(f"  Center Frequency: {params['center_frequency']} MHz")
#         print(f"  Chirp Rate: {params['chirp_rate']} MHz/us")
#         print(f"  Start Time Index: {params['start_time_index']}")
#         print(f"  End Time Index: {params['end_time_index']}")
#         print('---------------------------')

# # Now `clusters_info` contains all the cluster data organized by rangeline

