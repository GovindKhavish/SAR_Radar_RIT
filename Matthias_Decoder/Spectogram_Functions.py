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

# -------------------- Row Adaptive Thresholding on Raw I/Q Data -----------------------------#
def adaptive_threshold_row(row, factor=2):
    mean_value = np.mean(np.abs(row)) 
    std_value = np.std(np.abs(row)) 
    threshold = mean_value + factor * std_value
    thresholded_row = np.where(np.abs(row) < threshold, 0, row)
    return thresholded_row

# -------------------- Clustering using DBSCAN -----------------------------#
def count_chirps_with_clustering(thresholded_array, eps=1, min_samples=2):
    # Identify non-zero points
    non_zero_indices = np.where(thresholded_array > 0)[0]
    
    # If there are no non-zero points, return 0 chirps
    if len(non_zero_indices) == 0:
        return 0
    
    # Prepare the data for clustering (this could be time indices or frequency indices)
    # Here we're using the indices as 1D data for DBSCAN
    data = non_zero_indices.reshape(-1, 1)
    
    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    
    # DBSCAN labels (-1 means noise/outliers, other numbers represent clusters)
    labels = db.labels_
    
    # Count the number of unique clusters (ignoring noise)
    num_chirps = len(set(labels)) - (1 if -1 in labels else 0)
    
    return num_chirps

# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
def identify_clusters(row, max_gap=15, min_cluster_size=30):
    clusters = []
    current_cluster = []
    gap_count = 0

    for i, val in enumerate(row):
        if val != 0:
            current_cluster.append(i)
            gap_count = 0 
        elif current_cluster:
            gap_count += 1
            if gap_count > max_gap:
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = []
                gap_count = 0
                
    if current_cluster and len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    clustered_row = np.zeros_like(row)
    for cluster in clusters:
        for idx in cluster:
            clustered_row[idx] = row[idx]

    return clustered_row

# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
# Define the adaptive thresholding function
def adaptive_threshold(array, factor=2):
    mean_value = np.mean(array)
    std_value = np.std(array)
    
    # Compute the threshold as mean + factor * std
    threshold = mean_value + factor * std_value
    
    # Apply the thresholding
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

# -------------------- Feature Extraction -----------------------------#
# Step 6: Extract signal characteristics for each group
def extract_chirp_characteristics(aa_db_filtered, bb, cc):
    """Extract the characteristics of a single chirp"""
    # Dominant frequency
    dominant_frequencies = []
    for time_slice in aa_db_filtered.T:
        if np.any(time_slice > 0):
            dominant_freq_index = np.argmax(time_slice)
            dominant_frequencies.append(bb[dominant_freq_index])

    # Signal duration (time where there's any signal)
    signal_duration = np.count_nonzero(np.any(aa_db_filtered > 0, axis=0)) * (cc[1] - cc[0])

    # Frequencies above threshold
    freqs_above_threshold = bb[np.any(aa_db_filtered > 0, axis=1)]
    bandwidth = freqs_above_threshold.max() - freqs_above_threshold.min()

    # Peak frequency and time
    max_intensity_idx = np.unravel_index(np.argmax(aa_db_filtered), aa_db_filtered.shape)
    peak_frequency = bb[max_intensity_idx[0]]
    peak_time = cc[max_intensity_idx[1]]

    # Center frequency and chirp deviation
    center_frequency = (freqs_above_threshold.max() + freqs_above_threshold.min()) / 2
    chirp_deviation = freqs_above_threshold.max() - freqs_above_threshold.min()

    # Pulse width and chirp rate
    pulse_width = signal_duration
    chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us

    return {
        'signal_duration': signal_duration,
        'bandwidth': bandwidth,
        'peak_frequency': peak_frequency,
        'peak_time': peak_time,
        'center_frequency': center_frequency,
        'chirp_rate': chirp_rate
    }

def plot_chirp_groups(extracted_values):
    """Plot the extracted groups"""
    for group_index, entry in enumerate(extracted_values, start=1):
        group = entry['group']
        values_in_group = entry['values']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(values_in_group, marker='o', linestyle='-', label=f'Group: {group}')
        ax.set_title(f'Chirp {group_index}', fontweight='bold')
        ax.set_xlabel('Index', fontweight='bold')
        ax.set_ylabel('Extracted Value', fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

