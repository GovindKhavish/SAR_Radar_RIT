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
filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
#filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
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
    
    print(f"Number of detections: {hits}")
    return target_map


#------------------------ Apply CFAR filtering --------------------------------
# Spectrogram plot
idx_n = 36
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

# Process the data using the CFAR function
alarm_rate = 1e-9
num_guard_cells = 10
num_reference_cells = 5 
threshold_factor = set_alpha(2*num_reference_cells,alarm_rate)


fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'
aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
# Define the adaptive thresholding function
def adaptive_threshold(array, factor=2):
    mean_value = np.mean(array)
    std_value = np.std(array)
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

threshold,aa = adaptive_threshold(aa)

fig = plt.figure(12, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
dd = ax.imshow(10 * np.log10(aa), aspect='auto', origin='lower', cmap='Greys')
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Filtered Spectrogram (Threshold: {round(10*np.log10(threshold),2)} dB)', fontweight='bold')
plt.tight_layout()

# Radar data dimensions
time_size = aa.shape[1] # Freq
freq_size = aa.shape[0] # Time


# Create 2D Mask
#vert_guard,vert_avg,hori_Guard,hori_avg
vert_guard = 15
vert_avg = 30
hori_guard = 25
hori_avg = 30
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
print("Threshold Value: ",alpha)

# thres_map = cfar_method(aa,cfar_mask,alpha)
thres_map = cfar_method(aa,padded_mask,alpha)

# Plot the Threshold Map
plt.figure(figsize=(10, 5))
plt.imshow(thres_map, interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[0], bb[-1]])
plt.title('Threshold map')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()

# Detect the targets using the spectrogram data
aa_db_filtered = detect_targets(aa, thres_map)

# Plot the Target Map
plt.figure(figsize=(10, 5))
plt.imshow(np.flipud(aa_db_filtered), interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[-1], bb[0]])
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()

# ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

thresholded_aa_flat = aa_db_filtered .flatten()

# 2D (time, frequency)
time_freq_data = np.column_stack(np.where(aa_db_filtered > 0))  # Get non-zero points

# Frequency bins
frequency_indices = bb[time_freq_data[:, 0]]

# DBSCAN
dbscan = DBSCAN(eps=6, min_samples=20)
clusters = dbscan.fit_predict(time_freq_data)

# Plot threshold
fig_thresh = plt.figure(13, figsize=(6, 6), clear=True)
ax_thresh = fig_thresh.add_subplot(111)
aa_db_filtered = 10 * np.log10(aa_db_filtered  + 1e-10) 
dd = ax_thresh.imshow(aa_db_filtered, aspect='auto', origin='lower', cmap='Greys')



for i, cluster in enumerate(np.unique(clusters[clusters != -1])):  # Exclude noise here
    cluster_points = time_freq_data[clusters == cluster]
    ax_thresh.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i}', s=2)

cbar = plt.colorbar(dd, ax=ax_thresh)
cbar.set_label('Intensity [dB]')
ax_thresh.set_xlabel('Time [us]', fontweight='bold')
ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
ax_thresh.set_title(f'Filtered Spectrogram with DBSCAN (Threshold: dB)', fontweight='bold')
ax_thresh.legend()
plt.tight_layout()


# Number of clusters (excluding noise)
num_clusters = len(np.unique(clusters[clusters != -1]))
print(f"Number of clusters: {num_clusters}")

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