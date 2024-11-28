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
#---------------------------- CFAR Function -----------------------------#
def create_2d_mask(vertical_guard_cells, vertical_avg_cells, horizontal_guard_cells, horizontal_avg_cells):
    vertical_size = 2 * (vertical_guard_cells + vertical_avg_cells) + 1
    horizontal_size = 2 * (horizontal_guard_cells + horizontal_avg_cells) + 1
    
    mask = np.zeros((vertical_size, horizontal_size))
    
    center_row = vertical_guard_cells + vertical_avg_cells
    center_col = horizontal_guard_cells + horizontal_avg_cells

    total_avg_cells = (vertical_size*horizontal_size) - ((2*vertical_guard_cells+1)* (horizontal_guard_cells*2 +1))
    
    mask[:center_row - vertical_guard_cells, :] = 1/(total_avg_cells)
    mask[center_row + vertical_guard_cells + 1:, :] = 1/(total_avg_cells)
    mask[:, :center_col - horizontal_guard_cells] = 1/(total_avg_cells)
    mask[:, center_col + horizontal_guard_cells + 1:] = 1/(total_avg_cells)
    
    mask[center_row - vertical_guard_cells:center_row + vertical_guard_cells + 1, 
         center_col - horizontal_guard_cells:center_col + horizontal_guard_cells + 1] = 0
    
    return mask

def get_total_average_cells(vertical_guard_cells, vertical_avg_cells, horizontal_guard_cells, horizontal_avg_cells):
    vertical_size = 2 * (vertical_guard_cells + vertical_avg_cells) + 1
    horizontal_size = 2 * (horizontal_guard_cells + horizontal_avg_cells) + 1
    total_avg_cells = (vertical_size*horizontal_size) - ((2*vertical_guard_cells+1)* (horizontal_guard_cells*2 +1))
    #print(total_avg_cells)
    return total_avg_cells

### Padding ###

def create_2d_padded_mask(radar_data, cfar_mask):
    
    radar_rows, radar_cols = radar_data.shape
    mask_rows, mask_cols = cfar_mask.shape

    padded_mask = np.zeros((radar_rows, radar_cols))
    padded_mask[:mask_rows, :mask_cols] = cfar_mask

    return padded_mask

def set_alpha(total_avg_cells,alarm_rate):
    alpha = total_avg_cells*(alarm_rate**(-1/total_avg_cells)-1)
    return alpha

def cfar_method(radar_data, cfar_mask, threshold_multiplier):
    rows, cols = radar_data.shape
    threshold_map = np.zeros_like(radar_data)

    padded_mask = create_2d_padded_mask(radar_data,cfar_mask)

    fft_data = np.fft.fft2(radar_data)
    fft_mask = np.fft.fft2(padded_mask)
    
    fft_threshold = fft_data * fft_mask
    
    threshold_map = np.abs(np.fft.ifft2(np.fft.fftshift(fft_threshold)))
    threshold_map *= threshold_multiplier
    
    return threshold_map


### Detection ###
def detect_targets(radar_data, threshold_map):
    target_map = np.zeros_like(radar_data)
    len_row, len_col = radar_data.shape 

    hits = 0
    for row in range(len_row):
        for col in range(len_col):
            if np.abs(radar_data[row, col]) > threshold_map[row, col]:
                target_map[row, col] = 1
                hits += 1
            else:
                target_map[row, col] = 0
    
    return target_map


#------------------------ Apply CFAR filtering --------------------------------
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

global_pulse_number = 1

start_idx = 0 
end_idx = 100  
fs = 46918402.800000004  

global_cluster_params = {}

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

    #------------------------ Apply CFAR filtering --------------------------------
    # Process the data using the CFAR function
    alarm_rate = 1e-9
    num_guard_cells = 10
    num_reference_cells = 5 
    threshold_factor = set_alpha(2*num_reference_cells,alarm_rate)

    # Radar data dimensions
    #radar_data = radar_data[0:200,10000:15000]
    time_size = aa.shape[1] # Freq
    freq_size = aa.shape[0] # Time

    # Create 2D Mask
    #vert_guard,vert_avg,hori_Guard,hori_avg
    vert_guard = 15
    vert_avg = 20
    hori_guard = 25
    hori_avg = 20
    alarm_rate = 1e-9

    cfar_mask = create_2d_mask(vert_guard,vert_avg,hori_guard,hori_avg)

    # # Plot the CFAR Mask
    # plt.figure(figsize=(2, 10))
    # plt.imshow(cfar_mask, interpolation='none', aspect='auto')
    # plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
    # plt.xlabel('Fast Time')
    # plt.ylabel('Slow Time')
    # plt.colorbar(label='Filter Amplitude')

    padded_mask = create_2d_padded_mask(aa,cfar_mask)

    # Plot the Padded Mask
    # plt.figure(figsize=(2, 10))
    # plt.imshow(padded_mask, interpolation='none', aspect='auto')
    # plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
    # plt.xlabel('Fast Time')
    # plt.ylabel('Slow Time')
    # plt.colorbar(label='Filter Amplitude')

    alpha = set_alpha(get_total_average_cells(vert_guard,vert_avg,hori_guard,hori_avg),alarm_rate)

    # thres_map = cfar_method(aa,cfar_mask,alpha)
    thres_map = cfar_method(aa,padded_mask,alpha)


    # Detect the targets using the spectrogram data
    aa_db_filtered = detect_targets(aa, thres_map)


    # ------------------ DBSCAN Clustering -------------------
    thresholded_aa_flat = aa_db_filtered.flatten()

    time_freq_data = np.column_stack(np.where(aa_db_filtered > 0)) 
    frequency_indices = bb[time_freq_data[:, 0]]

    # DBSCAN
     # Check if targets are detected
    if time_freq_data.shape[0] == 0:
        print(f"No targets detected for rangeline {idx_n}. Exiting the loop.")
        continue  
    else: 
        dbscan = DBSCAN(eps=2, min_samples=10)
        clusters = dbscan.fit_predict(time_freq_data)

        num_clusters = len(np.unique(clusters)) - 1
        print(f"Number of clusters for rangeline {idx_n}: {num_clusters}")

        # ------------------ Skip Feature Extraction if More Than 2 Clusters -------------------
        if num_clusters > 2:
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

        cluster_time_indices = {}
        for (rangeline_idx, cluster_id), params_list in global_cluster_params.items():
            for params in params_list:
                start_time_idx = params['start_time_index']
                end_time_idx = params['end_time_index']
                iq_start_idx = spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
                iq_end_idx = spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
                cluster_time_indices[(rangeline_idx, cluster_id)] = (iq_start_idx, iq_end_idx)

        isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

        for idx in range(len(radar_section)):
            for (rangeline_idx, cluster_id), (iq_start_idx, iq_end_idx) in cluster_time_indices.items():
                if iq_start_idx <= idx <= iq_end_idx:
                    isolated_radar_data[idx] = radar_section[idx]
                    break  # Stop checking once matched


# print("Cluster Parameters for All Pulses:")
# pprint.pprint(global_cluster_params)




pulse_numbers = []
bandwidths = []
durations = []

for unique_key, params_list in global_cluster_params.items():
    for params in params_list:
        pulse_numbers.append(params['pulse_number'])
        bandwidths.append(params['bandwidth'])
        durations.append(params['pulse_duration'])

sorted_indices = np.argsort(pulse_numbers)
pulse_numbers = np.array(pulse_numbers)[sorted_indices]
bandwidths = np.array(bandwidths)[sorted_indices]
durations = np.array(durations)[sorted_indices]

plt.figure(figsize=(10, 5))
plt.plot(pulse_numbers, bandwidths, linestyle='-', color='b', label="Bandwidth")
plt.title("Pulse Number vs Bandwidth")
plt.xlabel("Pulse Number")
plt.ylabel("Bandwidth (MHz)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(pulse_numbers, durations, linestyle='-', color='g', label="Duration")
plt.title("Pulse Number vs Pulse Duration")
plt.xlabel("Pulse Number")
plt.ylabel("Pulse Duration (us)")
plt.grid(True)
plt.legend()
plt.show()