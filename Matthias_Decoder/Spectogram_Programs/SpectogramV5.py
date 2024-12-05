#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import pandas as pd
import os
import sqlite3
import numpy as np
import logging
import math
import cmath
import struct
import polars as pl
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter
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

# Mipur VH Filepath
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'
inputfile = filepath + filename

l0file = sentinel1decoder.Level0File(inputfile)

sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

headline = f'Sentinel-1 (burst {selected_burst}): '

radar_data = l0file.get_burst_data(selected_burst)

#------------------------ Apply CFAR filtering --------------------------------
global_pulse_number = 1

start_idx = 1428
end_idx = 1429
fs = 46918402.800000004  

global_cluster_params = {}
global_isolated_pulses_data = {}

for idx_n in range(start_idx, end_idx + 1):
    radar_section = radar_data[idx_n, :]
    slow_time_offset = idx_n / fs 

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

    # -------------------- Adaptive Threshold on Intensity Data -----------------------------#
    threshold,aa = Spectogram_FunctionsV3.adaptive_threshold(aa)

    #------------------------ Apply CFAR filtering --------------------------------
    # Process the data using the CFAR function
    alarm_rate = 1e-9
    num_guard_cells = 10
    num_reference_cells = 5 
    threshold_factor = Spectogram_FunctionsV3.set_alpha(2*num_reference_cells,alarm_rate)

    # Radar data dimensions
    #radar_data = radar_data[0:200,10000:15000]
    time_size = aa.shape[1] # Freq
    freq_size = aa.shape[0] # Time

    # Create 2D Mask
    #vert_guard,vert_avg,hori_Guard,hori_avg
    vert_guard = 15
    vert_avg = 30
    hori_guard = 25
    hori_avg = 30
    alarm_rate = 1e-9

    cfar_mask = Spectogram_FunctionsV3.create_2d_mask(vert_guard,vert_avg,hori_guard,hori_avg)

    padded_mask = Spectogram_FunctionsV3.create_2d_padded_mask(aa,cfar_mask)

    alpha = Spectogram_FunctionsV3.set_alpha(Spectogram_FunctionsV3.get_total_average_cells(vert_guard,vert_avg,hori_guard,hori_avg),alarm_rate)

    thres_map = Spectogram_FunctionsV3.cfar_method(aa,padded_mask,alpha)

    aa_db_filtered = Spectogram_FunctionsV3.detect_targets(aa, thres_map)

    # ------------------ DBSCAN Clustering -------------------
    thresholded_aa_flat = aa_db_filtered.flatten()

    time_freq_data = np.column_stack(np.where(aa_db_filtered > 0)) 
    frequency_indices = bb[time_freq_data[:, 0]]

    # DBSCAN
    if time_freq_data.shape[0] == 0:
        print(f"No targets detected for rangeline {idx_n}.")
        continue  
    else: 
        dbscan = DBSCAN(eps=6, min_samples=20)
        clusters = dbscan.fit_predict(time_freq_data)

        num_clusters = len(np.unique(clusters[clusters != -1]))
        print(f"Number of clusters for rangeline {idx_n}: {num_clusters}")

        # ------------------ Skip Feature Extraction if More Than 2 Clusters -------------------
        if (num_clusters > 2 or num_clusters == 0):
            print(f"Skipping feature extraction for rangeline {idx_n} due to more than 2 clusters.")
            continue

        # ------------------ Assign Global Pulse Numbers and Adjusted Times -------------------
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  # Noise
                cluster_points = time_freq_data[clusters == cluster_id]
                frequency_indices = bb[cluster_points[:, 0]]
                time_indices = cluster_points[:, 1]
                bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
                center_frequency = np.mean(frequency_indices)
                time_span = np.max(time_indices) - np.min(time_indices)
                chirp_rate = bandwidth / time_span if time_span != 0 else 0

                start_time = np.min(time_indices) / fs  
                end_time = np.max(time_indices) / fs  
                adjusted_start_time = start_time + slow_time_offset
                adjusted_end_time = end_time + slow_time_offset

                pulse_duration = adjusted_end_time - adjusted_start_time

                unique_key = (idx_n, cluster_id) 

                if unique_key not in global_cluster_params:
                    global_cluster_params[unique_key] = []

                global_cluster_params[unique_key].append({
                    'pulse_number': global_pulse_number,  
                    'bandwidth': bandwidth,
                    'center_frequency': center_frequency,
                    'chirp_rate': chirp_rate,
                    'start_time_index': np.min(time_indices),
                    'end_time_index': np.max(time_indices),
                    'adjusted_start_time': adjusted_start_time, 
                    'adjusted_end_time': adjusted_end_time,      
                    'pulse_duration': pulse_duration            
                })
                global_pulse_number += 1 

        # ------------------ Process I/Q Data -------------------
        NFFT = 256
        noverlap = 200
        sampling_rate = fs

        time_step = (NFFT - noverlap) / sampling_rate

        # Dictionary to store isolated I/Q data for each pulse (global storage)
        isolated_pulses_data = {}

        cluster_time_indices = {}
        for (rangeline_idx, cluster_id), params_list in global_cluster_params.items():
            for params in params_list:
                start_time_idx = params['start_time_index']
                end_time_idx = params['end_time_index']
                iq_start_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
                iq_end_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
                
                pulse_number = params['pulse_number']
                if pulse_number not in cluster_time_indices:
                    cluster_time_indices[pulse_number] = []
                cluster_time_indices[pulse_number].append((rangeline_idx, cluster_id, iq_start_idx, iq_end_idx))

        # Initialize isolated data for each pulse
        for pulse_number in cluster_time_indices:
            isolated_pulses_data[pulse_number] = []

        # Isolate radar data for each pulse
        for idx in range(len(radar_section)):
            for pulse_number, clusters in cluster_time_indices.items():
                for (rangeline_idx, cluster_id, iq_start_idx, iq_end_idx) in clusters:
                    if iq_start_idx <= idx <= iq_end_idx:
                        if len(isolated_pulses_data[pulse_number]) <= idx:
                            isolated_pulses_data[pulse_number].extend([0] * (idx - len(isolated_pulses_data[pulse_number]) + 1))
                        isolated_pulses_data[pulse_number][idx] = radar_section[idx]
                        break  # Stop checking once matched

        # Convert lists to numpy arrays for each pulse
        for pulse_number in isolated_pulses_data:
            isolated_pulses_data[pulse_number] = np.array(isolated_pulses_data[pulse_number], dtype=complex)

        # Update the global variable with isolated data for this rangeline (don't overwrite)
        for pulse_number, data in isolated_pulses_data.items():
            if pulse_number not in global_isolated_pulses_data:
                global_isolated_pulses_data[pulse_number] = []
            global_isolated_pulses_data[pulse_number].append(data)  # Append data from this rangeline









