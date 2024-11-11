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

radar_data_thresholded = np.array([Spectogram_Functions.adaptive_threshold_row(row, factor=2) for row in radar_data])

radar_data_clusters = np.array([Spectogram_Functions.identify_clusters(row, max_gap=10, min_cluster_size=30) for row in radar_data_thresholded])

plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(np.abs(radar_data_clusters)), aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Clustered Raw I/Q Data')
plt.show()

# User-defined start and end rows for processing
start_row = 1245  # Set this to the desired starting row index
end_row = 1250    # Set this to the desired ending row index (inclusive)
fs = 46918402.800000004  # in Hz
slow_time_interval = 1 / fs * 1e6  # Convert to microseconds

# Validate the start and end indices
if start_row < 0 or end_row >= len(radar_data):
    raise ValueError("Start and end rows must be within the bounds of the radar data.")
if start_row > end_row:
    raise ValueError("The start row must be less than or equal to the end row.")

all_rangeline_characteristics = []

# -------------------- Loop Through Specified Rangelines -----------------
for idx_n in range(start_row, end_row + 1):
    
    # Print progress
    print(f"Processing Rangeline {idx_n}/{end_row}")
    
    # Extract radar section for the current rangeline
    radar_section = radar_data_thresholded[idx_n, :]
    
    # Generate the spectrogram for the radar section
    fig = plt.figure(11, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(111)
    
    aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs / 1e6, Fc=None, detrend=None, window=np.hanning(256), scale='dB', noverlap=200, cmap='Greys')
    
    # cbar = plt.colorbar(dd, ax=ax)
    # cbar.set_label('Intensity [dB]')
    # ax.set_xlabel('Time [us]', fontweight='bold')
    # ax.set_ylabel('Freq [MHz]', fontweight='bold')
    # ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
    # plt.tight_layout()
    # plt.pause(0.1)

    # Apply adaptive threshold to the spectrogram
    threshold, aa_db_filtered = Spectogram_Functions.adaptive_threshold(aa, factor=2)

    # # Filtered spectrogram plot (optional)
    # fig = plt.figure(12, figsize=(6, 6), clear=True)
    # ax = fig.add_subplot(111)
    # dd = ax.imshow(10 * np.log10(aa_db_filtered), aspect='auto', origin='lower', cmap='Greys')
    
    # cbar = plt.colorbar(dd, ax=ax)
    # cbar.set_label('Intensity [dB]')
    # ax.set_xlabel('Time [us]', fontweight='bold')
    # ax.set_ylabel('Freq [MHz]', fontweight='bold')
    # ax.set_title(f'Filtered Spectrogram (Threshold: {round(10 * np.log10(threshold), 2)} dB)', fontweight='bold')
    # plt.tight_layout()
    # plt.pause(0.1)
    
# Assuming non_zero_indices and groups are returned from the Spectrogram Function
non_zero_indices, groups = Spectogram_Functions.group_consecutive_time_indices(aa_db_filtered)
characteristics = Spectogram_Functions.process_groups_and_extract_characteristics(groups, aa_db_filtered, bb, cc, non_zero_indices)

adjusted_groups = []
for group in groups:
    start_time = group[0] * slow_time_interval  
    end_time = group[-1] * slow_time_interval
    adjusted_groups.append([group, start_time, end_time])

# Append the characteristics of the current rangeline to the major storage variable
all_rangeline_characteristics.append([idx_n, characteristics, adjusted_groups])

# Accessing and printing the first rangeline characteristics
first_rangeline_data = all_rangeline_characteristics[0] 

# Print characteristics directly
for group_char in first_rangeline_data[1]:  # Access characteristics``
    for i in range(0,5):
        print(f"Group Characteristics: {group_char[i]}")
