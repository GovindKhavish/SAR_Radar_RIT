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


#----------------------------------Functions----------------------------------------------



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
filepath = "/Users/khavishgovind/Documents/Masters/Data/NorthernSea_Ireland/S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE/"
filename = 's1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat'


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
selected_burst  = 11;

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
RFI_slice = radar_data[136:137,450:459]
noise_slice = radar_data[137:138,:700]
Test_slice = radar_data[134:140,:700]
#print(radar_data[136:137,69:618])

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

### Plot Slices
# plt.plot(abs(radar_data[136:137,70:618]).reshape(-1),label='Interference')
# plt.plot(abs(radar_data[137:138,69:618]).reshape(-1),c = 'r',label='Noise')
# plt.legend(loc='upper right')
# plt.show()



# Calculate the minimum, maximum, and mean values
min_value = np.min(abs(radar_data[136:137,69:618]))
max_value = np.max(abs(radar_data[136:137,69:618]))
mean_value = np.mean(abs(radar_data[136:137,69:618]))
print(min_value)
print(max_value)
print(mean_value)


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

#plt.plot(abs(radar_data[136:137,69:618]).reshape(-1))
#plt.plot(abs(radar_data[137:138,69:618]).reshape(-1),c = 'r')
#plt.show()


#---------------------------- CFAR Function -----------------------------#
def create_cfar_filter(num_slow_time, num_fast_time, total_slow_time, total_fast_time):
    """
    Create a CFAR filter with averaging over the given dimensions.
    
    Parameters:
        num_slow_time (int): Number of slow time samples to include in the filter.
        num_fast_time (int): Number of fast time samples to include in the filter.
        total_slow_time (int): Total number of samples in the radar data.
        total_fast_time (int): Total number of fast time samples in the radar data.
        
    Returns:
        np.ndarray: Zero-padded CFAR filter.
    """
    # Initialize the CFAR filter with ones
    cfar_filter = np.ones((num_slow_time, num_fast_time))
    
    # Normalize the filter to have average values
    cfar_filter /= (num_slow_time * num_fast_time)
    
    # Zero-pad the filter to match the dimensions of the radar data FFT
    padded_filter = np.zeros((total_slow_time, total_fast_time))
    padded_filter[:num_slow_time, :num_fast_time] = cfar_filter

    return padded_filter


# def create_cfar_filter(total_slow_time, total_fast_time, num_guard_cells, num_averaging_cells, cut_position_slow=None):
#     """
#     Create a CFAR filter with specified number of averaging cells and guard cells around the CUT.
    
#     Parameters:
#         total_slow_time (int): Total number of slow time samples in the radar data.
#         total_fast_time (int): Total number of fast time samples in the radar data.
#         num_guard_cells (int): Total number of guard cells to be split evenly around the CUT.
#         num_averaging_cells (int): Number of averaging cells on either side of the guard cells.
#         cut_position_slow (int): Optional parameter to specify the slow time position of the CUT. If None, defaults to the center.
    
#     Returns:
#         np.ndarray: Zero-padded CFAR filter with CUT, guard cells, and averaging cells marked.
#     """
#     # Initialize the CFAR filter with zeros
#     padded_filter = np.zeros((total_slow_time, total_fast_time))
    
#     # Center or fixed position for the CUT
#     cut_slow = cut_position_slow if cut_position_slow is not None else total_slow_time // 2
#     cut_fast = total_fast_time // 2

#     # Number of guard cells on either side of the CUT
#     num_guard_side = num_guard_cells // 2
    
#     # Define the CUT
#     start_cut_fast = cut_fast
#     end_cut_fast = cut_fast + 1  # CUT is 1 fast time unit in width
#     padded_filter[cut_slow, start_cut_fast:end_cut_fast] = 50  # Mark CUT with a distinct value (e.g., 50)
    
#     # Define the guard cells
#     start_guard_fast = cut_fast - num_guard_side
#     end_guard_fast = cut_fast + num_guard_side + 1  # +1 to include CUT in guard cells
    
#     # Ensure guard cells are within bounds
#     start_guard_fast = max(start_guard_fast, 0)
#     end_guard_fast = min(end_guard_fast, total_fast_time)
    
#     # Set guard cells
#     padded_filter[cut_slow, start_guard_fast:end_guard_fast] = 1.5  # Mark guard cells with a distinct value (e.g., 1.5)
    
#     # Define the averaging cells
#     start_avg_fast = start_guard_fast - num_averaging_cells
#     end_avg_fast = end_guard_fast + num_averaging_cells
    
#     # Ensure averaging cells are within bounds
#     start_avg_fast = max(start_avg_fast, 0)
#     end_avg_fast = min(end_avg_fast, total_fast_time)
    
#     # Set averaging cells
#     padded_filter[cut_slow, start_avg_fast:start_guard_fast] = 1  # Averaging cells (left of guard cells)
#     padded_filter[cut_slow, end_guard_fast:end_avg_fast] = 1     # Averaging cells (right of guard cells)
    
#     return padded_filter



def apply_cfar_filter(num_slow_time, num_fast_time,radar_data, cfar_filter):
    """
    Apply the CFAR filter to the radar data using FFT and inverse FFT.
    
    Parameters:
        radar_data (np.ndarray): The raw radar data.
        cfar_filter (np.ndarray): The CFAR filter to apply.
        
    Returns:
        np.ndarray: The filtered radar data.
    """
    # FFT of the radar data
    radar_data_fft = np.fft.fft2(radar_data)
    
    # # Zero-pad the CFAR filter to match the dimensions of radar_data_fft
    # padded_filter = create_cfar_filter(radar_data.shape[0], radar_data.shape[1], 
    #                                    num_guard_cells=550, num_averaging_cells=100)
     # Zero-pad the CFAR filter to match the dimensions of radar_data_fft
    padded_filter = create_cfar_filter(num_slow_time,num_fast_time,radar_data.shape[0], radar_data.shape[1])
    
    # FFT of the CFAR filter
    filter_fft = np.fft.fft2(padded_filter)
    
    # Apply the filter in the frequency domain
    filtered_data_fft = radar_data_fft * filter_fft
    
    # Inverse FFT to get back to the time domain
    filtered_data = np.fft.ifft2(filtered_data_fft).real
    
    return filtered_data

#------------------------ Apply CFAR filtering --------------------------------

# Parameters for the CFAR filter
num_slow_time = 1
num_fast_time = 550
total_slow_time = Test_slice.shape[0]
total_fast_time = Test_slice.shape[1]
num_guard_cells = 550
num_averaging_cells = 100

# Create the CFAR filter
# cfar_filter = create_cfar_filter(total_slow_time, total_fast_time, num_guard_cells=num_guard_cells, num_averaging_cells=num_averaging_cells)

# Create the CFAR filter
cfar_filter = create_cfar_filter(num_slow_time, num_fast_time,total_slow_time, total_fast_time)

# Plot the CFAR filter
plt.figure(figsize=(10, 10))
plt.imshow(cfar_filter, interpolation='none', aspect='auto')
plt.title('CFAR Filter with CUT, Guard Cells, and Averaging Cells')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.colorbar(label='Filter Amplitude')
plt.show()

# Apply CFAR filter to the radar data
filtered_radar_data = apply_cfar_filter(num_slow_time,num_fast_time,Test_slice, cfar_filter)

# Compare original and filtered data point by point
comparison_data = np.where(Test_slice > filtered_radar_data, Test_slice, filtered_radar_data)

# Plot original, filtered, and comparison data
fig, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].imshow(np.abs(Test_slice), aspect='auto', interpolation='none', origin='lower')
ax[0].set_title('Original Radar Data', fontweight='bold')
ax[0].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[0].set_ylabel('Slow Time (cross range)', fontweight='bold')

ax[1].imshow(np.abs(filtered_radar_data), aspect='auto', interpolation='none', origin='lower')
ax[1].set_title('Filtered Radar Data', fontweight='bold')
ax[1].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[1].set_ylabel('Slow Time (cross range)', fontweight='bold')

ax[2].imshow(np.abs(comparison_data), aspect='auto', interpolation='none', origin='lower')
ax[2].set_title('Comparison Data', fontweight='bold')
ax[2].set_xlabel('Fast Time (down range)', fontweight='bold')
ax[2].set_ylabel('Slow Time (cross range)', fontweight='bold')

plt.tight_layout()
plt.show()





