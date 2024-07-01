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
# ... Ende der Import-Anweisungen


import sys
from pathlib import Path, PurePath;
##########################################################################################
###  eigene Bibliotheken aus den SubDirectories laden
_simraddir     = Path('sentinel1decoder');
if not _simraddir.exists():
    ###  Only strings should be added to sys.path
    sys.path.insert(0, Path().resolve().parent.__str__() );

import sentinel1decoder;


##########################################################################################
##########################################################################################
#filepath    = './data/RadarNorthOfIrland/';
#filename    = 's1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat';
##filename = 's1a-iw-raw-s-vv-20200705t181540-20200705t181612-033323-03dc5b.dat';

#filepath    = './data/Pakistan/';
##filename    = 's1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat';
#filename    = 's1a-iw-raw-s-vv-20220115t130440-20220115t130513-041472-04ee76.dat';


### -> https://nbviewer.org/github/Rich-Hall/sentinel1Level0DecodingDemo/blob/main/sentinel1Level0DecodingDemo.ipynb
filepath    = '/Users/khavishgovind/Documents/Masters/Data/Sao_Paulo/Sao_Paulo_HH_HV/S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE/'
filename    = 's1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat'  #-> Example from https://github.com/Rich-Hall/sentinel1decoder'
##filename    = 's1b-iw-raw-s-vv-20210216t083028-20210216t083100-025629-030def.dat';  #-> Example from 'https://github.com/jmfriedt/sentinel1_level0'

inputfile = filepath+filename

l0file = sentinel1decoder.Level0File(inputfile)

##############
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
print(bust_info)

### The satellite ephemeris data is sub-commutated across multiple packets due to its
###  relatively low update rate, so we need to perform an extra step to extract this information.
sent1_ephe  = l0file.ephemeris
#breakpoint()

##############
### 3 - Extract Data
# 3.1 - Select Packets to Process
# After extracted all the packet metadata the next step is to select the data packets we'll
# be processing. We want to exclude all packets that don't contain SAR instrument returns,
# and then pick a small set of these to operate on.


if filepath  == './data/RadarNorthOfIrland/':
    selected_burst  = 49;    ### for RadarNorthOfIrland
#
elif filepath == './data/SaoPaulo/':
    selected_burst = 8;     #-> Example from https://github.com/Rich-Hall/sentinel1decoder'
    #selected_burst = 9;     ### Blocks = 28874!
#
else:
    selected_burst  = 9;    ##7, 9, 11 könnten für Pakistan von Interesse sein
    #selected_burst  = 21;


print("Into selection mode")
print("Selected burst %f",selected_burst)

selection       = l0file.get_burst_metadata(selected_burst);
#selection

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
    selection       = l0file.get_burst_metadata(selected_burst);
    print(f'Not working with {selected_burst}. will try the next one!');

#breakpoint()
headline = f'Sentinel-1 (burst {selected_burst}): '
print(headline)


### 3.2 - Extract Raw I/Q Sensor Data
# Now we're ready to extract the raw sensor output from the file. The result will be a
# set of complex I/Q samples measured by the SAR instrument. By stacking these
# horizontally we can produce a 2D array of data samples, with fast_time along one axis
# and slow_time $\eta$ along the other. Since all the required information to do this
# is contained in packet metadata, the decoder outputs data arranged like this
# automatically.

### Decode the IQ data
#radar_data_raw = l0file.get_burst_data(selected_burst, try_load_from_file=False, save=False);
radar_data = l0file.get_burst_data(selected_burst);     ### liegen dann noch immer im speicher vor, auch wenn darauf gearbeitet wird


### Cache this data so we can retreive it more quickly next time we want it
#l0file.save_burst_data(selected_burst);


print("Starting to graph")
### Plotting our array, we can see that although there is clearly some structure to the
### data, we can't yet make out individual features. Our image needs to be focused along
### range and azimuth axes.
# Plot the raw IQ data extracted from the data file
fig = plt.figure(1, figsize=(8, 8), clear=True);
ax  = fig.add_subplot(111);
ax.imshow(np.abs(radar_data), aspect='auto', interpolation='none', vmin=0, vmax=15, origin='lower');
#ax.imshow(np.abs(radar_data_raw), aspect='auto', interpolation='none',); #origin='lower');
ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold');
ax.set_xlabel('Fast Time (down range)', fontweight='bold');
ax.set_ylabel('Slow Time (cross range)', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();