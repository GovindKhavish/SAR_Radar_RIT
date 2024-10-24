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

# Load the Level 0 File
l0file = sentinel1decoder.Level0File(inputfile)

# Extract Metadata and Burst Information
sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

# Select the Burst to Process
selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

headline = f'Sentinel-1 (burst {selected_burst}): '

# Extract Raw I/Q Sensor Data
radar_data = l0file.get_burst_data(selected_burst)

# # # Plotting the result
# plt.figure(figsize=(14, 6))
# plt.imshow(10*np.log10(abs(radar_data[:,:])), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.title('Orginal Data')


# -------------------- Coherent Compression -----------------------------#
# Parameters
idx_n = 1000
fs = 46918402.800000004

radar_section = radar_data[idx_n,:]#[:,idx_n]

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

scale = 'dB'
# aa - matrix containing intensity values for each time freqeuncy bin
# bb - Array for freunqecy values
# cc - Array of time segments
# dd - Image for matplotlib
aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
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
    
    # Compute the threshold as mean + factor * std
    threshold = mean_value + factor * std_value
    
    # Apply the thresholding
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

# Apply the adaptive thresholding to the intensity array `aa`
threshold,aa_db_filtered = adaptive_threshold(aa, factor=2)

# Plot the thresholded spectrogram
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
# -------------------- Chirp Segmentation -----------------------------#
# Get the indices of non-zero values in the thresholded spectrogram
non_zero_indices = np.nonzero(aa_db_filtered)

# Assuming non_zero_indices is obtained from the thresholded spectrogram
non_zero_indices = np.nonzero(aa_db_filtered)

# Convert the time indices (non_zero_indices[1]) to a list
time_indices = list(non_zero_indices[1])

# Sort the list and remove duplicates
unique_sorted_time_indices = sorted(set(time_indices))

# Initialize a list to store groups of consecutive indices
groups = []

# Track the current group and the last seen index
current_group = []
last_index = None

# Iterate through the unique sorted time indices
for index in unique_sorted_time_indices:
    # If this is the first index, start a new group
    if last_index is None:
        current_group.append(index)
    else:
        # Check for a break in consecutive values
        if index == last_index + 1:
            # Still in the same group
            current_group.append(index)
        else:
            # Break detected; save the current group and start a new one
            groups.append(current_group)
            current_group = [index]  # Start a new group

    # Update the last seen index
    last_index = index

# Don't forget to add the last group to the list
if current_group:
    groups.append(current_group)

# Output the groups of consecutive time indices
print("Number of Groups:\n")
print(len(groups))
print("Groups of consecutive time indices:")

if(len(groups) == 1):
    non_zero_indices = np.nonzero(aa_db_filtered)
    bb_non_zero = bb[non_zero_indices[1]]
    cc_non_zero = cc[non_zero_indices[0]] 

    # -------------------- Extract Characteristics -----------------------------#
    # dominant frequency
    dominant_frequencies = []
    for time_slice in aa_db_filtered.T:
        if np.any(time_slice > 0): 
            dominant_freq_index = np.argmax(time_slice)
            dominant_frequencies.append(bb[dominant_freq_index])  

    # signal duration
    signal_duration = np.count_nonzero(np.any(aa_db_filtered > 0, axis=0)) * (cc[1] - cc[0])

    # bandwidth
    freqs_above_threshold = bb[np.any(aa_db_filtered > 0, axis=1)]  # Frequencies
    bandwidth = freqs_above_threshold.max() - freqs_above_threshold.min()

    # peak frequency
    max_intensity_idx = np.unravel_index(np.argmax(aa_db_filtered), aa_db_filtered.shape)
    peak_frequency = bb[max_intensity_idx[0]]  # Frequency 
    peak_time = cc[max_intensity_idx[1]]       # Time 

    # Center frequency (using midpoint between highest and lowest frequencies)
    center_frequency = (freqs_above_threshold.max() + freqs_above_threshold.min()) / 2

    # Chirp Deviation
    chirp_deviation = freqs_above_threshold.max() - freqs_above_threshold.min()

    # Chirp Rate (using the new formula)
    pulse_width = signal_duration  # Signal duration as pulse width
    chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us

    # Output the results
    #print(f"Dominant Frequencies: {dominant_frequencies}")
    print(f"\nSignal Duration: {round(signal_duration,3)} us")
    print(f"Bandwidth: {round(bandwidth,3)} MHz")
    print(f"Peak Frequency: {round(peak_frequency,3)} MHz at time {round(peak_time,3)} us")
    print(f"Center Frequency: {round(center_frequency,3)} MHz")  # Updated center frequency calculation
    print(f"Average Chirp Rate: {round(chirp_rate, 3)} MHz/us\n")
    plt.show()

else:
    extracted_values = []

    # Iterate through each group
    for group in groups:
        # Extract the indices from non_zero_indices corresponding to the current group
        values_in_group = non_zero_indices[0][np.isin(non_zero_indices[1], group)]
        
        # Store the extracted values
        extracted_values.append({
            'group': group,
            'values': values_in_group
        })

    # Iterate through each extracted group to plot the values
    for group_index, entry in enumerate(extracted_values, start=1):  # Start counting from 1
        group = entry['group']  # Current group of time indices
        values_in_group = entry['values']  # Corresponding values from non_zero_indices

        # Create a new figure for the current group
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the values in the group
        ax.plot(values_in_group, marker='o', linestyle='-', label=f'Group: {group}')

        # Enhance the plot
        ax.set_title(f'Chirp {group_index}', fontweight='bold')
        ax.set_xlabel('Index', fontweight='bold')
        ax.set_ylabel('Extracted Value', fontweight='bold')
        ax.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def extract_sig_characteristics(extracted_values, time_array):
        # Initialize variables
        durations = []
        bandwidths = []
        peak_frequencies = []
        center_frequencies = []
        chirp_rates = []
        
        for entry in extracted_values:
            group = entry['group']
            values = entry['values']
            
            # Get the corresponding times and frequencies
            time_indices = np.array(group)  # Indices of time
            frequencies = time_array[values]  # Frequencies corresponding to these indices

            # Calculate Duration
            duration = time_indices[-1] - time_indices[0]  # Time duration in microseconds
            durations.append(duration)
            
            # Calculate Bandwidth
            bandwidth = frequencies.max() - frequencies.min()  # Bandwidth in MHz
            bandwidths.append(bandwidth)
            
            # Calculate Peak Frequency
            peak_frequency = frequencies[np.argmax(values)]  # Frequency corresponding to the highest value
            peak_frequencies.append(peak_frequency)
            
            # Calculate Center Frequency
            center_frequency = (frequencies.max() + frequencies.min()) / 2  # Center frequency
            center_frequencies.append(center_frequency)

            # Calculate Chirp Deviation
            chirp_deviation = frequencies.max() - frequencies.min()  # Difference between max and min frequency

            # Calculate Pulse Width
            pulse_width = duration  # Pulse width is the duration calculated earlier

            # Calculate Chirp Rate using the formula
            chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us
            chirp_rates.append(chirp_rate)
        # Return all calculated metrics
        return {
            'durations': durations,
            'bandwidths': bandwidths,
            'peak_frequencies': peak_frequencies,
            'center_frequencies': center_frequencies,
            'chirp_rates': chirp_rates
        }

    # Example usage to print metrics for each group
    characteristics_v3 = extract_sig_characteristics(extracted_values, bb)

    # Print each metric per group
    for i in range(len(characteristics_v3['durations'])):
        print(f"Chirp {i + 1}:")
        print(f"  Duration: {round(characteristics_v3['durations'][i], 3)} us")
        print(f"  Bandwidth: {round(characteristics_v3['bandwidths'][i], 3)} MHz")
        print(f"  Peak Frequency: {round(characteristics_v3['peak_frequencies'][i], 3)} MHz")
        print(f"  Center Frequency: {round(characteristics_v3['center_frequencies'][i], 3)} MHz")
        print(f"  Chirp Rate: {round(characteristics_v3['chirp_rates'][i], 3)} MHz/us")
        print("\n")  # For better separation between groups


