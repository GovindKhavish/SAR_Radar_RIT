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
import Spectogram_Functions
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
from scipy.signal import windows
from scipy.signal import butter, filtfilt
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path, PurePath;
#-----------------------------------------------------------------------------------------
# Define the subdirectory path
_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

# Check if the subdirectory exists
if _simraddir.exists():
    # Add the subdirectory to sys.path
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder;

#-----------------------------------------------------------------------------------------
### -> https://nbviewer.org/github/Rich-Hall/sentinel1Level0DecodingDemo/blob/main/sentinel1Level0DecodingDemo.ipynb

# Sao Paulo HH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\SaoPaulo_Brazil\HH\S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE"
#filename    = '\s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat'  #-> Example from https://github.com/Rich-Hall/sentinel1decoder'

# Sao Paulo VH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\SaoPaulo_Brazil\VH\S1B_IW_RAW__0SDV_20210216T083028_20210216T083100_025629_030DEF_1684.SAFE"
#filename    = '\s1b-iw-raw-s-vh-20210216t083028-20210216t083100-025629-030def.dat'

# New York HH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\NewYork_USA\S1A_IW_RAW__0SDH_20240610T105749_20240610T105815_054260_069997_EFB1.SAFE"
#filename = '\s1a-iw-raw-s-hh-20240610t105749-20240610t105815-054260-069997.dat'

# Dimona VH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\Dimona_Isreal\S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE"
#filename = '\s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat'

# Augsburg VH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\Augsburg_Germany\S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE"
#filename = '\s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat'

# Northern Sea VH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\NorthernSea_Ireland\S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE"
# filepath = "/Users/khavishgovind/Documents/Masters/Data/NorthernSea_Ireland/S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE/"
# filename = 's1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat'

# White Sands VH
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\WhiteSand_USA\S1A_IW_RAW__0SDV_20211214T130351_20211214T130423_041005_04DEF2_011D.SAFE\S1A_IW_RAW__0SDV_20211214T130351_20211214T130423_041005_04DEF2_011D.SAFE"
#filename = '\s1a-iw-raw-s-vh-20211214t130351-20211214t130423-041005-04def2.dat'

# Mipur VH
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

inputfile = filepath+filename

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

# plt.figure(figsize=(14, 6))
# plt.imshow(10*np.log10(abs(radar_data[:,:])), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.title('Orginal Data')


radar_data_thresholded = np.array([Spectogram_Functions.adaptive_threshold_row(row, factor=2) for row in radar_data])

radar_data_clusters = np.array([Spectogram_Functions.identify_clusters(row, max_gap=10, min_cluster_size=30) for row in radar_data_thresholded])

plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(np.abs(radar_data_clusters)), aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Clustered Raw I/Q Data')
plt.show()

# -------------------- Spectogram used -----------------------------#
# Parameters
idx_n = 1400
fs = 46918402.800000004

radar_section = radar_data_thresholded[idx_n,:]#[:,idx_n]

# fig = plt.figure(10, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)
# ax.plot(np.abs(radar_section), label=f'abs{idx_n}')
# ax.plot(np.real(radar_section), label=f'Re{idx_n}')
# ax.plot(np.imag(radar_section), label=f'Im{idx_n}')
# ax.legend()
# ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold')
# ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold')
# ax.set_ylabel('|Amplitude|', fontweight='bold')
# plt.tight_layout()
# plt.pause(0.1)

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)

# aa - matrix containing intensity values for each time freqeuncy bin
# bb - Array for freunqecy values
# cc - Array of time segments
# dd - Image for matplotlib
aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale='dB',noverlap=200, cmap='Greys')
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

threshold,aa_db_filtered = Spectogram_Functions.adaptive_threshold(aa, factor=2)

fig = plt.figure(12, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
dd = ax.imshow(10 * np.log10(aa_db_filtered), aspect='auto', origin='lower', cmap='Greys')
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Filtered Spectrogram (Threshold: {round(10*np.log10(threshold),2)} dB)', fontweight='bold')
plt.tight_layout()
plt.show()

# Step 1: Find non-zero values in the spectrogram
non_zero_indices = np.nonzero(aa_db_filtered)

# Step 2: Extract time indices (column indices of non-zero values)
time_indices = list(non_zero_indices[1])
unique_sorted_time_indices = sorted(set(time_indices))

# Step 3: Group consecutive time indices
groups = []
current_group = []
last_index = None

# Iterate over the unique time indices
for index in unique_sorted_time_indices:
    if last_index is None:
        current_group.append(index)
    else:
        if index == last_index + 1:  # If consecutive index
            current_group.append(index)
        else:
            groups.append(current_group)  # Finalize the previous group
            current_group = [index]  # Start a new group

    last_index = index

# Add the last group if any
if current_group:
    groups.append(current_group)

# Check if the number of groups is more than 2
if len(groups) > 2:
    print("More than 2 groups detected. Skipping processing.")
else:
    print(f"Number of Groups of consecutive time indices: {len(groups)}\n")

    # For single group (if there's only one group, you process as before)
    if len(groups) == 1:
        non_zero_indices = np.nonzero(aa_db_filtered)
        
        bb_non_zero = bb[non_zero_indices[1]]
        cc_non_zero = cc[non_zero_indices[0]]

        # Extract signal characteristics for the single group
        dominant_frequencies = []
        for time_slice in aa_db_filtered.T:
            if np.any(time_slice > 0):
                dominant_freq_index = np.argmax(time_slice)
                dominant_frequencies.append(bb[dominant_freq_index])

        # Calculate signal duration, bandwidth, etc.
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

        # Output results for single group
        print(f"\nSignal Duration: {round(signal_duration, 3)} us")
        print(f"Bandwidth: {round(bandwidth, 3)} MHz")
        print(f"Peak Frequency: {round(peak_frequency, 3)} MHz at time {round(peak_time, 3)} us")
        print(f"Center Frequency: {round(center_frequency, 3)} MHz")
        print(f"Average Chirp Rate: {round(chirp_rate, 3)} MHz/us\n")
        plt.show()

    else:
        extracted_values = []
        for group in groups:
            values_in_group = non_zero_indices[0][np.isin(non_zero_indices[1], group)]
            extracted_values.append({'group': group, 'values': values_in_group})

        # Now we process each group and extract features using `extract_chirp_characteristics`
        characteristics_list = []

        # For each extracted group, create a sub-spectrogram and extract its characteristics
        for group_index, entry in enumerate(extracted_values, start=1):
            group = entry['group']
            values_in_group = entry['values']

            # Create a sub-spectrogram for the current group of values
            aa_db_filtered_group = np.zeros_like(aa_db_filtered)  # Create an empty matrix
            aa_db_filtered_group[non_zero_indices[0][np.isin(non_zero_indices[1], group)], non_zero_indices[1][np.isin(non_zero_indices[1], group)]] = aa_db_filtered[non_zero_indices[0][np.isin(non_zero_indices[1], group)], non_zero_indices[1][np.isin(non_zero_indices[1], group)]]  # Copy values to new matrix

            # Extract the characteristics for the current group
            group_characteristics = Spectogram_Functions.extract_chirp_characteristics(aa_db_filtered_group, bb, cc)
            characteristics_list.append(group_characteristics)

            # Optionally: Print the extracted characteristics for each group
            print(f"Chirp {group_index} Characteristics:")
            print(f"  Signal Duration: {round(group_characteristics['signal_duration'], 3)} us")
            print(f"  Bandwidth: {round(group_characteristics['bandwidth'], 3)} MHz")
            print(f"  Peak Frequency: {round(group_characteristics['peak_frequency'], 3)} MHz at time {round(group_characteristics['peak_time'], 3)} us")
            print(f"  Center Frequency: {round(group_characteristics['center_frequency'], 3)} MHz")
            print(f"  Chirp Rate: {round(group_characteristics['chirp_rate'], 3)} MHz/us")
            print("\n")

        # Plot the groups
        Spectogram_Functions.plot_chirp_groups(extracted_values)