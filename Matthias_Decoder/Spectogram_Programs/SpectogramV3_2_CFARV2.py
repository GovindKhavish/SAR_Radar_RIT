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
    
    threshold_map = np.abs(np.fft.ifft2(fft_threshold))
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
idx_n = 1070
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

# Radar data dimensions
#radar_data = radar_data[0:200,10000:15000]
time_size = aa.shape[1] # Freq
freq_size = aa.shape[0] # Time

# Create vertical CFAR mask
# cfar_mask = create_vert_mask(slow_time_size)
# padded_mask = create_padded_mask(radar_data,cfar_mask,1)

# Create horizontal CFAR mask
#cfar_mask = create_hori_mask(fast_time_size)
#padded_mask = create_padded_mask(radar_data,cfar_mask,1)

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
targets = detect_targets(aa, thres_map)

# Plot the Target Map
plt.figure(figsize=(10, 5))
plt.imshow(targets, interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[0], bb[-1]])
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()
plt.show()



