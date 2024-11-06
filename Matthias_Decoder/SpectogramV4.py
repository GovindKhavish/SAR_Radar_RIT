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
filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
#filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
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

plt.figure(figsize=(14, 6))
plt.imshow(10*np.log10(abs(radar_data[:,:])), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
plt.colorbar(label='Amplitude')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Orginal Data')

# -------------------- Row Adaptive Thresholding on Raw I/Q Data -----------------------------#
def adaptive_threshold_row(row, factor=2):
    mean_value = np.mean(np.abs(row)) 
    std_value = np.std(np.abs(row)) 
    threshold = mean_value + factor * std_value
    thresholded_row = np.where(np.abs(row) < threshold, 0, row)
    return thresholded_row

radar_data_thresholded = np.array([adaptive_threshold_row(row, factor=2) for row in radar_data])

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

radar_data_clusters = np.array([identify_clusters(row, max_gap=10, min_cluster_size=30) for row in radar_data_thresholded])

plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(np.abs(radar_data_clusters)), aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Clustered Raw I/Q Data')
plt.show()

# Save
# np.save('clustered_radar_data.npy', radar_data_clusters)
# loaded_data = np.load('clustered_radar_data.npy')

# -------------------- Coherent Compression -----------------------------#
# Parameters
idx_n = 240
fs = 46918402.800000004

spectrograms = []

for radar_section in radar_data_thresholded:
    aa, bb, cc, _ = plt.specgram(radar_section, NFFT=256, Fs=fs / 1e6, detrend=None,window=np.hanning(256), scale='dB', noverlap=200, cmap='Greys')
    spectrograms.append({'intensity': aa,'frequency': bb,'time': cc})

# -------------------- Adaptive Threshold Function -----------------------------#
# Define the adaptive thresholding function
def adaptive_threshold(array, factor=2):
    mean_value = np.mean(array)
    std_value = np.std(array)
    
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

# -------------------- Row-wise Feature Extrcation -----------------------------#
time_interval = 1 / fs
all_characteristics = []

for row_index, spectrogram_data in enumerate(spectrograms):
    aa = spectrogram_data['intensity']
    bb = spectrogram_data['frequency']
    cc = spectrogram_data['time'] + row_index * time_interval 

    # -------------------- Adaptive Threshold on Intensity Data -----------------------------
    threshold, aa_db_filtered = adaptive_threshold(aa, factor=2)

    # -------------------- Chirp Segmentation -----------------------------
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
                current_group = [index]  # New group
        last_index = index

    if current_group:
        groups.append(current_group)

    characteristics = []

    if len(groups) == 1:
        non_zero_indices = np.nonzero(aa_db_filtered)
        bb_non_zero = bb[non_zero_indices[1]]
        cc_non_zero = cc[non_zero_indices[0]]

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

        characteristics.append({
            'duration': signal_duration,
            'bandwidth': bandwidth,
            'peak_frequency': peak_frequency,
            'center_frequency': center_frequency,
            'chirp_rate': chirp_rate
        })

    else:
        extracted_values = []

        for group in groups:
            values_in_group = non_zero_indices[0][np.isin(non_zero_indices[1], group)]
            extracted_values.append({'group': group, 'values': values_in_group})

        def extract_sig_characteristics(extracted_values, time_array):
            durations = []
            bandwidths = []
            peak_frequencies = []
            center_frequencies = []
            chirp_rates = []
            
            for entry in extracted_values:
                group = entry['group']
                values = entry['values']
                
                time_indices = np.array(group)
                frequencies = time_array[values] 

                duration = time_indices[-1] - time_indices[0]
                durations.append(duration)
                
                bandwidth = frequencies.max() - frequencies.min()  # MHz
                bandwidths.append(bandwidth)
                
                peak_frequency = frequencies[np.argmax(values)]
                peak_frequencies.append(peak_frequency)
                
                center_frequency = (frequencies.max() + frequencies.min()) / 2
                center_frequencies.append(center_frequency)

                chirp_deviation = frequencies.max() - frequencies.min()
                pulse_width = duration
                chirp_rate = chirp_deviation / pulse_width if pulse_width != 0 else 0  # MHz/us
                chirp_rates.append(chirp_rate)

            return {
                'durations': durations,
                'bandwidths': bandwidths,
                'peak_frequencies': peak_frequencies,
                'center_frequencies': center_frequencies,
                'chirp_rates': chirp_rates
            }

        # Get characteristics for each chirp in the current spectrogram
        characteristics = extract_sig_characteristics(extracted_values, bb)

    # Store the characteristics of the current spectrogram
    all_characteristics.append(characteristics)
