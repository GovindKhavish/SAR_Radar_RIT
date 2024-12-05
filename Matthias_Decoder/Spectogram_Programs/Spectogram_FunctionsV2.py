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
from scipy.signal import butter, filtfilt
from sklearn.cluster import DBSCAN

# ------------------ Spectrogram Plot with Local Adaptive Thresholding -------------------
def adaptive_threshold_local(spectrogram_data, threshold_factor):
    mean_val = np.mean(spectrogram_data)
    std_val = np.std(spectrogram_data)
    threshold = mean_val + threshold_factor * std_val

    thresholded_data = np.where(spectrogram_data > threshold, spectrogram_data, 0)
    
    return threshold,thresholded_data

# ------------------ Spectrogram DBSCAN -------------------
def apply_dbscan(aa_db_filtered, bb, eps, min_samples):

    # Flatten the array 
    # Identify non-zero points for clustering
    time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points
    print("Max index in time_freq_data[:, 0]:", time_freq_data[:, 0].max())
    print("Size of bb:", len(bb))
    # Ensure we're only selecting valid frequency bins (0 to 255)
    valid_indices = time_freq_data[:, 0] < len(bb)
    time_freq_data = time_freq_data[valid_indices]  # Filter out invalid time-frequency pairs

    # Alternatively, clamp the values in time_freq_data[:, 0] to fit within the valid range for frequency bins
    clamped_indices = np.clip(time_freq_data[:, 0], 0, len(bb) - 1)

    # Use the clamped indices to access valid frequency bins
    frequency_indices = bb[clamped_indices]

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(time_freq_data)

    return clusters, frequency_indices, time_freq_data

# ------------------ Extract Parameters -------------------
def extract_cluster_params(time_freq_data, clusters, bb):
    # ------------------ Start and End Time for Each Cluster -------------------
    cluster_time_indices = {}

    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Exclude noise points (-1)
            # Time-freq points for the current cluster
            cluster_points = time_freq_data[clusters == cluster_id]
            # Time indices (second column)
            time_indices = cluster_points[:, 1]
            
            # Start and End time indices
            start_time_index = np.min(time_indices)
            end_time_index = np.max(time_indices)

            cluster_time_indices[cluster_id] = (start_time_index, end_time_index)
    
    # ------------------ Extract Cluster Parameters -------------------
    cluster_params = {}

    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Exclude noise points (-1)
            # Get the time-frequency points corresponding to the current cluster
            cluster_points = time_freq_data[clusters == cluster_id]
            
            # Frequency indices (mapped to the frequency bins `bb`)
            frequency_indices = bb[cluster_points[:, 0]]
            
            # Time indices (second column of time_freq_data)
            time_indices = cluster_points[:, 1]
            
            # Bandwidth: Difference between the maximum and minimum frequency values
            bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
            
            # Center frequency: Average frequency within the cluster
            center_frequency = np.mean(frequency_indices)
            
            # Chirp rate: Bandwidth divided by time span (approximated)
            time_span = np.max(time_indices) - np.min(time_indices)
            if time_span != 0:
                chirp_rate = bandwidth / time_span  # MHz per microsecond
            else:
                chirp_rate = 0  # In case time span is zero (unlikely)
            
            # Store the cluster parameters
            cluster_params[cluster_id] = {
                'bandwidth': bandwidth,
                'center_frequency': center_frequency,
                'chirp_rate': chirp_rate,
                'start_time_index': np.min(time_indices),
                'end_time_index': np.max(time_indices)
            }
    
    return cluster_time_indices, cluster_params

# ------------------ Clusters to IQ Data -------------------

def map_clusters_to_iq_indices(cluster_params, fs, radar_section, NFFT=256, noverlap=200):

    # Time steps
    time_step = (NFFT - noverlap) / fs  # Time per specto bin (seconds)

    def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
        return (time_indices * time_step * sampling_rate).astype(int)

    mapped_cluster_indices = {}
    for cluster_id, params in cluster_params.items():
        start_time_idx = params['start_time_index']
        end_time_idx = params['end_time_index']
        iq_start_idx = spectrogram_to_iq_indices(start_time_idx, fs, time_step)
        iq_end_idx = spectrogram_to_iq_indices(end_time_idx, fs, time_step)
        mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

    isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

    for idx in range(len(radar_section)):
        for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
            if iq_start_idx <= idx <= iq_end_idx:
                isolated_radar_data[idx] = radar_section[idx]
                break  

    return isolated_radar_data


