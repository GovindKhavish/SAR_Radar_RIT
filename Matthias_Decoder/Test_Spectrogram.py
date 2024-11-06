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
selected_burst = 63
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

# Extract Raw I/Q Sensor Data
radar_data = l0file.get_burst_data(selected_burst)


# -------------------- Coherent Compression -----------------------------#
# Parameters
range_bin = 1500
num_bins=10

def coherent_compression(radar_data, range_bin, num_bins):
    if range_bin - num_bins < 0 or range_bin + num_bins >= radar_data.shape[1]:
        raise ValueError("Range bin with specified number of bins is out of bounds.")

    len_row = radar_data.shape[0]
    compressed_radar_data = np.zeros((len_row, 1),dtype=complex)

    for row_idx in range(len_row):

        start_idx = range_bin - num_bins
        end_idx = range_bin + num_bins + 1

        sum_values = np.sum(radar_data[row_idx, start_idx:end_idx])
        compressed_radar_data[row_idx, 0] = sum_values

    return compressed_radar_data

compressed_data = coherent_compression(radar_data, range_bin,num_bins)

# Plotting the result
plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# #plt.plot(np.abs(compressed_data))
# plt.imshow(10*np.log10(np.abs(compressed_data)), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.title(f'Coherently Compressed Data for Range Bins {range_bin-num_bins} to {range_bin+num_bins}')
# plt.ylabel('Slow Time')
# plt.xlabel('Fast Time')

# plt.subplot(1, 2, 2)
plt.imshow(10*np.log10(abs(radar_data[:,:])), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
plt.colorbar(label='Amplitude')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Orginal Data')

#plt.show()


# -------------------- Add Spectrogram Generation -----------------------#
# Parameters
window_size = 10
padding_size = 128 - window_size
PRI = 0.00275

# Calculate Spectrogram for each coloumn (Range bin) of the radar_data
# radar_data[slow time, fast time]


def create_spectrogram(radar_data, range_bin, window_size, PRI, num_bins, padding_size, overlap=0.50):

    #compressed_data = coherent_compression(radar_data, range_bin, num_bins)
    compressed_data = radar_data

    spectrogram = []
    
    shift = int(np.floor(window_size * (1 - overlap)))
    if shift < 1:
        shift = 1

    #hamming_window = np.hamming(window_size * 2)

    for i in range(window_size, compressed_data.shape[0] - window_size + 1, shift):
        window_data = compressed_data[i - window_size:i + window_size, 0]
        window_data = window_data #* hamming_window 
        padded_window_data = np.concatenate((window_data, np.zeros(padding_size)))  
        fft_result = np.fft.fftshift(np.fft.fft(padded_window_data))  
        spectrogram.append(np.abs(fft_result)) 
    
    spectrogram = np.array(spectrogram).T
    
    num_points = spectrogram.shape[1]
    total_time = (num_points - 1) * shift * PRI
    t_axis = np.linspace(0, total_time, num_points)

    return spectrogram, t_axis

spectrogram, t_axis = create_spectrogram(radar_data, range_bin, window_size, PRI,num_bins,padding_size)

# # # Plotting
plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.plot(np.arange(len(radar_data[:, range_bin])) * PRI, np.abs(radar_data[:, range_bin]))
# plt.title(f'Original Radar Data for Range Bin {range_bin}')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

#plt.subplot(1, 2, 2)
plt.imshow(10*np.log10(spectrogram), aspect='auto', extent=[t_axis[0], t_axis[-1],0, spectrogram.shape[1]], origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram for Range Bin {range_bin}')

# # Plot the spectrogram using imshow
# plt.figure(figsize=(10, 6))
# plt.imshow(abs(radar_data[:,range_bin:range_bin+1]), aspect='auto', interpolation='none', origin='lower',vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.xlabel('Tast Time')
# plt.ylabel('Slow Time')
# plt.title(f'Spectrogram for Range Bin {range_bin}')

plt.tight_layout()
plt.show()



