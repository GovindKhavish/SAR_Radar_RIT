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

# Inline CFAR function
def set_alpha(total_avg_cells,alarm_rate):
    alpha = total_avg_cells*(alarm_rate**(-1/total_avg_cells)-1)
    return alpha

def cfar_1d(data, num_guard_cells, num_reference_cells, threshold_factor):

    thresholded_data = np.zeros_like(data)
    num_cells = len(data)
    
    for i in range(num_reference_cells + num_guard_cells, num_cells - (num_reference_cells + num_guard_cells)):
        lagging_cells = data[i - num_reference_cells - num_guard_cells:i - num_guard_cells]
        leading_cells = data[i + num_guard_cells + 1:i + num_guard_cells + num_reference_cells + 1]
        
        noise_level = np.sum(np.abs(np.concatenate((leading_cells, lagging_cells)))) / (2 * num_reference_cells)
        thresholded_data[i] = np.abs(threshold_factor * noise_level)
        # print('\n')
        # print(np.abs(data[i]))
        # print(np.abs(threshold_factor * noise_level))

        # if np.abs(data[i]) > np.abs(threshold_factor * noise_level):
        #     thresholded_data[i] = 1#data[i]
        # else:
        #     thresholded_data[i] = 0
    
    return thresholded_data

def cfar_detector(iq_data, guard_cells, training_cells, pfa):
    # Calculate the amplitude of IQ data
    amplitude = np.abs(iq_data)
    num_cells = len(amplitude)
    
    total_training_cells = 2 * training_cells
    alpha = total_training_cells * (pfa ** (-1 / total_training_cells) - 1)
    print(alpha)

    # Initialize an empty list for detected peaks
    detected_peaks = []

    # Slide across the data to compute the threshold for each CUT
    for cut_idx in range(training_cells + guard_cells, num_cells - training_cells - guard_cells):

        leading_train = amplitude[cut_idx - training_cells - guard_cells : cut_idx - guard_cells]
        trailing_train = amplitude[cut_idx + guard_cells + 1 : cut_idx + guard_cells + 1 + training_cells]

        noise_level = (np.sum(leading_train) + np.sum(trailing_train)) / (total_training_cells)
        
        threshold = alpha * noise_level
        
        # Check if the CUT exceeds the threshold
        if amplitude[cut_idx] > threshold:
            detected_peaks.append(1)
        else:
            detected_peaks.append(0)

    return np.array(detected_peaks)


# Spectrogram plot
idx_n = 1070
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

# Process the data using the CFAR function
alarm_rate = 1e-9
num_guard_cells = 100
num_reference_cells = 1000 
threshold_factor = set_alpha(2*num_reference_cells,alarm_rate)
print(threshold_factor)
#adar_data_thresholded = cfar_1d(radar_section, num_guard_cells, num_reference_cells, threshold_factor)
radar_data_thresholded = cfar_detector(radar_section, num_guard_cells, num_reference_cells, alarm_rate)


fig = plt.figure(10, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
#ax.plot(np.abs(radar_data_thresholded), label=f'abs{idx_n}')
ax.plot(np.real(radar_data_thresholded), label=f'Re{idx_n}')
#ax.plot(np.imag(radar_data_thresholded), label=f'Im{idx_n}')
ax.legend()
ax.set_title(f'{headline} Thresholded', fontweight='bold')
ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold')
ax.set_ylabel('|Amplitude|', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
ax.plot(np.abs(radar_section), label=f'abs{idx_n}')
#ax.plot(np.real(radar_section), label=f'Re{idx_n}')
#ax.plot(np.imag(radar_section), label=f'Im{idx_n}')
ax.legend()
ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold')
ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold')
ax.set_ylabel('|Amplitude|', fontweight='bold')
plt.tight_layout()
plt.show()

# fig = plt.figure(11, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)
# aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6, detrend=None, window=np.hanning(256), scale='dB', noverlap=200, cmap='Greys')
# cbar = plt.colorbar(dd, ax=ax)
# cbar.set_label('Intensity [dB]')
# ax.set_xlabel('Time [us]', fontweight='bold')
# ax.set_ylabel('Freq [MHz]', fontweight='bold')
# ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
# plt.tight_layout()
# plt.pause(0.1)

# # Inline adaptive threshold for spectrogram
# def adaptive_threshold(matrix, factor=2):
#     mean = np.mean(matrix)
#     std_dev = np.std(matrix)
#     threshold = mean + factor * std_dev
#     return threshold, np.where(matrix > threshold, matrix, 0)

# threshold, aa_db_filtered = adaptive_threshold(aa, factor=2)

# fig = plt.figure(12, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)
# dd = ax.imshow(10 * np.log10(aa_db_filtered), aspect='auto', origin='lower', cmap='Greys')
# cbar = plt.colorbar(dd, ax=ax)
# cbar.set_label('Intensity [dB]')
# ax.set_xlabel('Time [us]', fontweight='bold')
# ax.set_ylabel('Freq [MHz]', fontweight='bold')
# ax.set_title(f'Filtered Spectrogram (Threshold: {round(10*np.log10(threshold), 2)} dB)', fontweight='bold')
# plt.tight_layout()
# plt.show()

