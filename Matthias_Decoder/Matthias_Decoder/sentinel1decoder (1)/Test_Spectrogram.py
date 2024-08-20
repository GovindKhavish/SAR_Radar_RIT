##CFar_Test

#!/usr/bin/env python
# -*- coding: utf-8 -*-
### -*- coding: iso-8859-15 -*-
""" Auswertung Sentianl 1 Raw Data

:Info:
    Version: 2024.05
    Author : Matthias Weiß
"""
#use this function from one folder up so the import will work :-)

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
filename = '\s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

inputfile = filepath+filename

# Load the Level 0 File
l0file = sentinel1decoder.Level0File(inputfile)

# Extract Metadata and Burst Information
sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

# Select the Burst to Process
selected_burst = 63
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

# Extract Raw I/Q Sensor Data
radar_data = l0file.get_burst_data(selected_burst)



# -------------------- Add Spectrogram Generation -----------------------#
# Calculate Spectrogram for each row (Slow Time) of the radar_data
# radar_data[slow time, fast time]

def create_spectrogram(radar_data, range_bin, window_size, PRI):
    if range_bin >= radar_data.shape[1]:
        raise ValueError(f"range_bin {range_bin} is out of bounds for radar_data with {radar_data.shape[1]} fast time bins.")

    spectrogram = []
    t_axis = []
    padding_size = 122

    window_length = window_size + padding_size
    hann_window = np.hanning(window_size*2) 

    for i in range(window_size,radar_data.shape[0] - window_size + 1):
        window_data = radar_data[i-window_size:i + window_size, range_bin]
        window_data = window_data * hann_window
        padded_window_data = np.concatenate((window_data, np.zeros(padding_size)))
        fft_result = np.fft.fft(padded_window_data)
        spectrogram.append(np.abs(fft_result))

    spectrogram = np.array(spectrogram).T 
    
    num_points = spectrogram.shape[1]  
    total_time = (num_points - 1) * PRI  
    t_axis = np.linspace(0, total_time, num_points)

    return spectrogram, t_axis

range_bin = 10
window_size = 3
PRI = 0.00275 # 2.75 miliseconds

spectrogram, t_axis = create_spectrogram(radar_data, range_bin, window_size,PRI)

# Plot the spectrogram using imshow
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram, aspect='auto', extent=[t_axis[0], t_axis[-1],0, spectrogram.shape[1]], origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram for Range Bin {range_bin}')

# # Plot the spectrogram using imshow
# plt.figure(figsize=(10, 6))
# plt.imshow(abs(radar_data[:,range_bin:range_bin+1]), aspect='auto', interpolation='none', origin='lower',vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.xlabel('Time (s)')
# plt.ylabel('Slow Time (cross range')
# plt.title(f'Spectrogram for Range Bin {range_bin}')

# Plot the original radar data and the spectrogram side by side
plt.figure(figsize=(14, 6))

# Plot the original radar data
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(radar_data[:, range_bin])) * PRI, np.abs(radar_data[:, range_bin]))
plt.title('Original Radar Data for Range Bin 10')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the spectrogram using imshow
plt.subplot(1, 2, 2)
plt.imshow(spectrogram, aspect='auto', extent=[t_axis[0], t_axis[-1],0, spectrogram.shape[1]], origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram for Range Bin {range_bin}')

plt.tight_layout()
plt.show()

# # Select one row (range line) from the 2D radar data
# selected_row_index = 179  # Change this to the index of the range line you want to display
# range_line = radar_data[selected_row_index, :]

# # Compute the spectrogram for the selected range line
# frequencies, times, Sxx = spectrogram(range_line, fs=1.0, nperseg=256)

# # Create a figure with three subplots side by side
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), clear=True)

# # Plot the original radar data
# cax = ax1.imshow(abs(radar_data[selected_row_index-5:selected_row_index+5,:]), aspect='auto', interpolation='none', origin='lower',vmin=0,vmax=10)
# ax1.set_title('Original Radar Data', fontweight='bold')
# ax1.set_xlabel('Fast Time (down range)', fontweight='bold')
# ax1.set_ylabel('Slow Time (cross range)', fontweight='bold')
# plt.colorbar(cax, ax=ax1, label='Magnitude')

# # Plot the selected range line from the radar data using imshow
# ax2.imshow(abs(range_line)[np.newaxis, :], aspect='auto', origin='lower')
# ax2.set_title(f'Range Line {selected_row_index} from Radar Data', fontweight='bold')
# ax2.set_xlabel('Fast Time (down range)', fontweight='bold')
# ax2.set_ylabel('Amplitude', fontweight='bold')
# ax2.set_xticks([])  # Remove x ticks
# ax2.set_yticks([])  # Remove y ticks

# # Plot the spectrogram of the selected range line
# im = ax3.imshow(10 * np.log10(Sxx), aspect='auto', interpolation='none', origin='lower', 
#                 extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
# ax3.set_title('Spectrogram of Selected Range Line', fontweight='bold')
# ax3.set_xlabel('Time [s]', fontweight='bold')
# ax3.set_ylabel('Frequency [Hz]', fontweight='bold')
# plt.colorbar(im, ax=ax3, label='Intensity [dB]')

# # Adjust layout and display the plot
# plt.tight_layout()
# plt.pause(0.1)
# plt.show()


