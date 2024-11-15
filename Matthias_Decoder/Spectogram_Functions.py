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
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

# -------------------- Feature Extraction -----------------------------#
def extract_chirp_characteristics(aa_db_filtered, bb, cc):
    dominant_frequencies = []
    for time_slice in aa_db_filtered.T:
        if np.any(time_slice > 0):
            dominant_freq_index = np.argmax(time_slice)
            dominant_frequencies.append(bb[dominant_freq_index])

    # Signal duration
    signal_duration = np.count_nonzero(np.any(aa_db_filtered > 0, axis=0)) * (cc[1] - cc[0])

    freqs_above_threshold = bb[np.any(aa_db_filtered > 0, axis=1)]
    bandwidth = freqs_above_threshold.max() - freqs_above_threshold.min()

    # Peak frequency and Peak time
    max_intensity_idx = np.unravel_index(np.argmax(aa_db_filtered), aa_db_filtered.shape)
    peak_frequency = bb[max_intensity_idx[0]]
    peak_time = cc[max_intensity_idx[1]]

    # Center frequency
    center_frequency = (freqs_above_threshold.max() + freqs_above_threshold.min()) / 2
    chirp_deviation = freqs_above_threshold.max() - freqs_above_threshold.min()

    # Pulse width and Chirp rate
    pulse_width = signal_duration
    chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us

    # Start and End times
    non_zero_columns = np.any(aa_db_filtered > 0, axis=0) 
    start_time = cc[np.argmax(non_zero_columns)] 
    stop_time = cc[len(non_zero_columns) - 1 - np.argmax(np.flip(non_zero_columns))]  

    return [
        signal_duration,
        bandwidth,
        peak_frequency,
        peak_time,
        center_frequency,
        chirp_rate,
        start_time,
        stop_time
    ]

# -------------------- Plotting -----------------------------#
def plot_chirp_groups(extracted_values):
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

# -------------------- Group Seperation -----------------------------#
def group_consecutive_time_indices(aa_db_filtered):

    non_zero_indices = np.nonzero(aa_db_filtered)
    time_indices = list(non_zero_indices[1])
    unique_sorted_time_indices = sorted(set(time_indices))

    groups = []
    current_group = []
    last_index = None

    for index in unique_sorted_time_indices:
        if last_index is None:
            current_group.append(index)
        else:
            if index == last_index + 1: 
                current_group.append(index)
            else:
                groups.append(current_group) 
                current_group = [index] 

        last_index = index

    if current_group:
        groups.append(current_group)

    return non_zero_indices, groups

def process_groups_and_extract_characteristics(groups, aa_db_filtered, bb, cc, non_zero_indices):
    characteristics_list = []

    if len(groups) > 2:
        print("\nMore than 2 groups detected. Skipping processing.")
    else:
        #print(f"Number of Groups of consecutive time indices: {len(groups)}\n")

        if len(groups) == 1:
            non_zero_indices = np.nonzero(aa_db_filtered)

            dominant_frequencies = []
            for time_slice in aa_db_filtered.T:
                if np.any(time_slice > 0):
                    dominant_freq_index = np.argmax(time_slice)
                    dominant_frequencies.append(bb[dominant_freq_index])

            signal_duration = np.count_nonzero(np.any(aa_db_filtered > 0, axis=0)) * (cc[1] - cc[0])
            freqs_above_threshold = bb[np.any(aa_db_filtered > 0, axis=1)]
            bandwidth = freqs_above_threshold.max() - freqs_above_threshold.min()

            max_intensity_idx = np.unravel_index(np.argmax(aa_db_filtered), aa_db_filtered.shape)
            peak_frequency = bb[max_intensity_idx[0]]
            peak_time = cc[max_intensity_idx[1]]

            center_frequency = (freqs_above_threshold.max() + freqs_above_threshold.min()) / 2
            chirp_deviation = freqs_above_threshold.max() - freqs_above_threshold.min()

            pulse_width = signal_duration
            chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us

            start_time = cc[np.min(non_zero_indices[1])]
            stop_time = cc[np.max(non_zero_indices[1])]

            group_characteristics = [
                signal_duration,
                bandwidth,
                peak_frequency,
                peak_time,
                center_frequency,
                chirp_rate,
                start_time,
                stop_time
            ]
            characteristics_list.append(group_characteristics)

        else:
            extracted_values = []
            for group in groups:
                values_in_group = non_zero_indices[0][np.isin(non_zero_indices[1], group)]
                extracted_values.append([group, values_in_group])

            for entry in extracted_values:
                group = entry[0]
                values_in_group = entry[1]

                aa_db_filtered_group = np.zeros_like(aa_db_filtered)
                aa_db_filtered_group[non_zero_indices[0][np.isin(non_zero_indices[1], group)], non_zero_indices[1][np.isin(non_zero_indices[1], group)]] = aa_db_filtered[non_zero_indices[0][np.isin(non_zero_indices[1], group)], non_zero_indices[1][np.isin(non_zero_indices[1], group)]]  # Copy values to new matrix

                group_characteristics = extract_chirp_characteristics(aa_db_filtered_group, bb, cc)
                characteristics_list.append(group_characteristics)

    return characteristics_list
