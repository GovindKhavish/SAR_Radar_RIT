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
from scipy.interpolate import interp1d
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
filename = '\s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'


inputfile = filepath+filename

l0file = sentinel1decoder.Level0File(inputfile)

#-----------------------------------------------------------------------------------------
"""
2 - Extract File Metadata

Sentinel-1 level 0 data files consist of the raw packetized data sent to the ground.
One packet typically consists of the radar instrument output associated with one radar
echo, so a single file typically consists of many thousands of packets. Packets may also
consist of other types of data e.g. background noise measurements for instrument
calibration.

We are working with data acquired in stripmap mode over Sao Paulo and the nearby port of
Santos, Brazil. Stripmap data is mainly used to monitor small islands, so is relatively
infrequently used. However, it is relatively simple compared to Interferometric Wide Swath
mode, the main acquisiton mode used over land, and therefore makes our task of image
formation much simpler!

Initially we're going to pull the metadata from each packet and output to a Pandas
dataframe to examine the file contents. Producing an image from the entirity of the data
found in a single file would take a long time and require a lot of memory, so we're
aiming to produce an image from just a subset of this data.
"""

sent1_meta  = l0file.packet_metadata
bust_info   = l0file.burst_info


#-------------------------CSV Method for Indexes ------------------------#
#bust_info.to_csv('NorthernSea(Ireland)_VH_Burst_Index.csv', index=False)

# The satellite ephemeris data is sub-commutated across multiple packets due to its
sent1_ephe  = l0file.ephemeris

#------------------------- 3. Extract Data -----------------------------#
### 3.1 - Select Packets to Process
selected_burst  = 57;

print("Into selection mode\n")

selection = l0file.get_burst_metadata(selected_burst);

while selection['Signal Type'].unique()[0] != 0:
    """ SIGTYPcode SIGTYP Description Notes
        0 Echo Radar echo signal (nominal SAR imaging)
        1 Noise Noise measurement
        2 to 7 - not applicable
        8 Tx Cal
        9 Rx Cal
        10 EPDN Cal
        11 TA Cal
        12 APDN Cal
        13 to 14 - not applicable
        15 TxH Cal Iso Tx Cal Isolation at Tx-Polarisation H
    """
    selected_burst +=1;
    selection = l0file.get_burst_metadata(selected_burst);
    print(f'Not working with {selected_burst}. will try the next one!');

headline = f'Sentinel-1 (burst {selected_burst}): '
print(headline)


### 3.2 - Extract Raw I/Q Sensor Data
### Decode the IQ data
radar_data = l0file.get_burst_data(selected_burst); 


### Format the complex numbers as strings
#formatted_data = np.array([[f"{x.real}+{x.imag}j" for x in row] for row in radar_data])
# csv_filename = 'radar_data_formatted_Ireland.csv'
# np.savetxt(csv_filename, formatted_data, delimiter=',', fmt='%s')
# print(f"Radar data has been saved to {csv_filename}")


### Cache this data so we can retreive it more quickly next time we want it
#l0file.save_burst_data(selected_burst);

print("Starting to graph")

### Plotting our array, we can see that although there is clearly some structure to the data
# Plot the raw IQ data extracted from the data file
# NOTE: The Fast axis is plotted on the x-axis but is is the coloumns in the radar_data

# fig = plt.figure(1, figsize=(8, 8), clear=True);
# ax  = fig.add_subplot(111);
# ax.imshow(np.abs(Test_slice), aspect='auto', interpolation='none', origin='lower');
# #ax.imshow(np.abs(radar_data_raw), aspect='auto', interpolation='none',); #origin='lower');
# ax.set_title(f'Sea North of Ireland- {headline} Raw I/Q Sensor Output', fontweight='bold');
# ax.set_xlabel('Fast Time (down range)', fontweight='bold');
# ax.set_ylabel('Slow Time (cross range)', fontweight='bold');
# plt.tight_layout(); plt.pause(0.1); plt.show();

### Pulse Amplitude Graphing
#pulse_amp = np.sqrt((np.real(RFI_slice)**2) + (np.imag(RFI_slice)**2))
#plt.plot(pulse_amp.reshape(-1))
#plt.show()


### Phase Modulation Graphing
#phase_mod = np.arctan(np.imag(radar_data[136:137,:700])/np.real(radar_data[136:137,:700]))
# plt.plot(phase_mod.reshape(-1))
# plt.show()


# plt.plot(abs(radar_data[136:137,:700]).reshape(-1),label='Interference')
# plt.plot(abs(radar_data[137:138,:700]).reshape(-1),c = 'r',label='Noise')
# plt.show()


#-------------------- 4. Convert data to 2D frequency domain-----------------------#
# FFT entire matrix
#radar_data = np.fft.fft2(radar_data)

# Shift zero frequency component to the center
#radar_data = np.fft.fftshift(radar_data)

# FFT for eeach range line (FastTime)
#radar_data = np.fft.fft(radar_data, axis=1);

# FFT each azimuth line (SlowTime) 
#radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0);

### Plot Frequnecy Domain
# fig = plt.figure(1, figsize=(8, 8), clear=True);
# ax  = fig.add_subplot(111);
# #ax.imshow(np.abs(radar_data), aspect='auto', interpolation='none', vmin=0, vmax=5 ,origin='lower');
# im = ax.imshow(np.abs(Test_slice), aspect='auto', interpolation='none', vmin=min_value, vmax=max_value ,origin='lower');
# #im = ax.imshow(np.abs(radar_data), aspect='auto', interpolation='none' ,origin='lower');
# ax.set_title(f'Sea North of Ireland- {headline} FFT Data', fontweight='bold');
# ax.set_xlabel('Fast Range Frequnecy (Hz)', fontweight='bold');
# ax.set_ylabel('Azimuth Frequnecy (Hz)', fontweight='bold');
# fig.colorbar(im, ax=ax)
# plt.tight_layout(); plt.pause(0.1); plt.show();

#---------------------------- CFAR Function -----------------------------#
def create_mask(size, num_guard_cells=3, num_averaging_cells=10):

    mask = np.zeros((size, 1))
    center = num_guard_cells + num_averaging_cells
    
    mask[center - num_guard_cells - num_averaging_cells:center - num_guard_cells] = 1
    mask[center + num_guard_cells + 1:center + num_guard_cells + 1 + num_averaging_cells] = 1
    
    mask /= (2 * num_averaging_cells)
    
    return mask

def create_padded_mask(radar_data,cfar_mask):
    rows = radar_data.shape[0]
    cols = radar_data.shape[1]

    temp_mask = np.zeros((rows,cols))
    temp_mask[:cfar_mask.shape[0], 0:cfar_mask.shape[1]] = cfar_mask
    return temp_mask
    
def cfar_1d(radar_data, cfar_mask, threshold_multiplier=1.5):
    rows, cols = radar_data.shape
    threshold_map = np.zeros_like(radar_data)

    padded_mask = create_padded_mask(radar_data,cfar_mask)

    fft_data = np.fft.fft2(radar_data)
    fft_mask = np.fft.fft2(padded_mask)
    
    fft_threshold = fft_data * fft_mask
    
    threshold_map = np.real(np.fft.ifft2(fft_threshold))
    
    threshold_map *= threshold_multiplier
    
    return threshold_map

def detect_targets(radar_data, threshold_map):
    target_map = np.zeros_like(radar_data)
    len_col = radar_data.shape[1]
    len_row = radar_data.shape[0]

    for row in range(len_row):

        for col in range(len_col):

            if(radar_data[row,col] > threshold_map[row,col]):
                target_map[row,col] = radar_data[row,col]

    return target_map


#------------------------ Apply CFAR filtering --------------------------------

# Example radar data dimensions (assuming radar_data is already defined)
fast_time_size = radar_data.shape[1]  # Extract the fast time dimension
slow_time_size = radar_data.shape[0]  # Extract the slow time dimension

# Create vertical CFAR mask
cfar_mask = create_mask(slow_time_size)
padded_mask = create_padded_mask(radar_data,cfar_mask)
thres_map = cfar_1d(radar_data,cfar_mask)
targets = detect_targets(radar_data, thres_map)

# # Plot the CFAR Mask
# plt.figure(figsize=(2, 10))
# plt.imshow(targets, interpolation='none', aspect='auto')
# plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.colorbar(label='Filter Amplitude')
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].imshow(abs(radar_data), aspect='auto', interpolation='none', origin='lower')
ax[0].set_title('Original Radar Data', fontweight='bold')
ax[0].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[0].set_ylabel('Slow Time (cross range)', fontweight='bold')

ax[1].imshow(np.abs(thres_map), aspect='auto', interpolation='none', origin='lower')
ax[1].set_title('Threshold Values', fontweight='bold')
ax[1].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[1].set_ylabel('Slow Time (cross range)', fontweight='bold')

ax[2].imshow(np.abs(targets), aspect='auto', interpolation='none', origin='lower')
ax[2].set_title('After CFAR', fontweight='bold')
ax[2].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[2].set_ylabel('Slow Time (cross range)', fontweight='bold')

plt.tight_layout()
plt.show()




