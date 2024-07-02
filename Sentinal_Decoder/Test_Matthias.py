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

# Sao Paulo HH
filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\SaoPaulo_Brazil\HH\S1A_S3_RAW__0SDH_20230518T213602_20230518T213627_048593_05D835_F012.SAFE"
filename    = '\s1a-s3-raw-s-hh-20230518t213602-20230518t213627-048593-05d835.dat'  #-> Example from https://github.com/Rich-Hall/sentinel1decoder'

# Sao Paulo VH
#filepath = r"C:\Users\govin\OneDrive\Desktop\Masters\Data\S1B_IW_RAW__0SDV_20210216T083028_20210216T083100_025629_030DEF_1684.SAFE"
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

bust_info.to_csv('output_SaoPaulo_HH.csv', index=False)

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



selected_burst  = 8;


print("Into selection mode")
print("Selected burst %f",selected_burst)

selection = l0file.get_burst_metadata(selected_burst);
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
    selection = l0file.get_burst_metadata(selected_burst);
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

### 4 - Image Processing
# The following section demonstrates an implementation of the range-Doppler algorithm. This
# essentially consists of the following steps:
#  - Range compression
#  - Transform to range-Doppler domain
#  - Range Cell Migration Correction (RCMC)
#  - Azimuth compression
#  - Transform to time domain
#  - Image formation

### 4.1 - Define auxiliary parameters
# We require a number of parameters in the calculations that. These are:
#  - Image sizes
#  - Various transmitted pulse parameters used to synthesize a replica Tx pulse
#  - Sample rates in range and azimuth
#  - The fast time $\tau$ associated with each range sample along a range line, and the corresponding slant range of closest approach $R_{0}$ for each of these range samples
#  - The frequency axes in range $f_{\tau}$ and azimuth $f_{\eta}$ after transforming our array to the frequency domain
#  - The effective spacecraft velocity $V_{r}$. This is a psuedo velocity approximated by $V_{r} \approx \sqrt{V_{s} V_{g}}$, where $V_{s}$ is the norm of the satellite velocity vector, and $V_{g}$ is the antenna beam velocity projected onto the ground. $V_{g}$ is calculated numerically acording to the method defined in https://iopscience.iop.org/article/10.1088/1757-899X/1172/1/012012/pdf. Note that $V_{g}$ and hence $V_{r}$ varies by slant range.
#  - The cosine of the instantaneous squint angle $D(f_{\eta}, V_{r})$, where
#
# $$D(f_{\eta}, V_{r}) = \sqrt{1 - \frac{c^{2} f_{\eta}^{2}}{4 V_{r}^{2} f_{0}^{2}}}$$

### Image sizes
len_range_line      = radar_data.shape[1];
len_az_line         = radar_data.shape[0]


### Tx pulse parameters
c0                  = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS;
RGDEC               = selection["Range Decimation"].unique()[0];
PRI                 = selection["PRI"].unique()[0];
rank                = selection["Rank"].unique()[0];
suppressed_data_time= 320/(8*sentinel1decoder.constants.F_REF);
range_start_time    = selection["SWST"].unique()[0] + suppressed_data_time;
wavelength          = sentinel1decoder.constants.TX_WAVELENGTH_M;
f0                  = c0/wavelength;

### Sample rates
range_sample_freq   = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC);
range_sample_period = 1 / range_sample_freq;
az_sample_freq      = 1 / PRI;
az_sample_period    = PRI;

### Create replica pulse
TXPSF               = selection['Tx Pulse Start Frequency'].unique()[0];
TXPRR               = selection['Tx Ramp Rate'].unique()[0];
TXPL                = selection['Tx Pulse Length'].unique()[0];
num_tx_vals         = int(TXPL * range_sample_freq);
tx_replica_time_vals= np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
phi1                = TXPSF + TXPRR*TXPL/2;
phi2                = TXPRR / 2.0;
tx_replica          = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2));


### Fast time vector ->  defines the time axis along the fast time direction
samples_along_range = np.arange(0, len_range_line, 1);
fast_time_vec       = range_start_time + (range_sample_period * samples_along_range);

### Slant range vector -> defines R0, the range of closest approach, for each range cell
slant_range_vec     = ((rank * PRI) + fast_time_vec) * c0/2;

### Axes -> defines the frequency axes in each direction after FFT
SWL                 = len_range_line / range_sample_freq;
az_freq_vals        = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line));
range_freq_vals     = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL);


### Spacecraft velocity -> numerical calculation of the effective spacecraft velocity
ecef_vels       = l0file.ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 + x["Y-axis velocity ECEF"]**2 +x["Z-axis velocity ECEF"]**2), axis=1)
velocity_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(), fill_value="extrapolate")
x_interp        = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["X-axis position ECEF"].unique(), fill_value="extrapolate")
y_interp        = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Y-axis position ECEF"].unique(), fill_value="extrapolate")
z_interp        = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Z-axis position ECEF"].unique(), fill_value="extrapolate")
space_velocities= selection.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float);

### Spacecraft Position (absolut position Earth centered)
x_positions     = selection.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
y_positions     = selection.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
z_positions     = selection.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
position_array  = np.transpose(np.vstack((x_positions, y_positions, z_positions)))

### actual earth radius using WGS84
a               = sentinel1decoder.constants.WGS84_SEMI_MAJOR_AXIS_M;
b               = sentinel1decoder.constants.WGS84_SEMI_MINOR_AXIS_M;
H               = np.linalg.norm(position_array, axis=1);
W               = np.divide(space_velocities, H);
lat             = np.arctan(np.divide(position_array[:, 2], position_array[:, 0]));
local_earth_rad = np.sqrt( np.divide( (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))),
                                      (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat)))
                                    ));
cos_beta        = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(slant_range_vec[:, np.newaxis]),
                             2 * local_earth_rad * H));
ground_velocities = local_earth_rad * W * cos_beta;
effective_velocities = np.sqrt(space_velocities * ground_velocities);

D               = np.sqrt( 1 - np.divide(wavelength**2 * np.square(az_freq_vals[:effective_velocities.shape[1]]),
                                         4.0 * np.square(effective_velocities) ) ).T;


### antenna beam pattern in azimuth
def antgain_phi(phi):
    antgain_phi     = np.sinc( sentinel1decoder.constants.ANTENNA_LENGTH /  wavelength * np.sin( phi ))**2;
    return( antgain_phi )
# -> phi_3dB = 0.004 [rad] -> beamwidth r_beam = 2 * r0 * sin(phi_3dB) ~ r0 * phi_3dB
# r_beam = 3744m  @  r0=936208m


print(f'Burst={selected_burst}: ');#'  \n ECC: {sentinel1decoder.constants.ECC_CODE[selection["ECC Number"].unique()[0]]}  Swath Number: {selection["Swath Number"].unique()[0]}   cos_beta={np.rad2deg(cos_beta[0,0]):.3f} -> 90-cos_beta={90-np.rad2deg(cos_beta[0,0]):.3f}')
print(f'    ECC_code= {sentinel1decoder.constants.ECC_CODE[selection["ECC Number"].unique()[0]]}');
print(f'    Swath Number= {selection["Swath Number"].unique()[0]}');
print(f'    cos_beta= {np.rad2deg(cos_beta[0,0]):.3f}  ->  90-cos_beta= {90-np.rad2deg(cos_beta[0,0]):.3f}');
#
### We're only interested in keeping D, so free up some memory by deleting these large arrays.
del effective_velocities, ground_velocities, cos_beta, local_earth_rad, a, b, H, W, lat
del position_array, x_positions, y_positions, z_positions


#breakpoint()
##############
### 4.2 - Convert data to 2D frequency domain
# We're going to be doing almost all our calculations in the frequency domain, so the
# first step is to FFT along the azimuth and range axes.

#FIXME: ab hier wird zuviel Speicher benötigt, der nicht vorhanden ist!!!
### FFT for eeach range line (FastTime)
radar_data = np.fft.fft(radar_data, axis=1);

###FIXME:  FFT each azimuth line (SlowTime) !!! anfalscher Stelle
#radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0);


#from scipy.fftpack import fft, ifft, fftshift
#radar_data_mf_f = fft(radar_data_raw, axis=1);


### 4.3 - Range compression -> create and apply matched filter
# Range compression is relatively simple. Range information is encoded in the arrival
# time of the pulse echo (i.e. an echo from a target further away will take longer to
# arrive), so by applying a matched filter consisting of the transmitted pulse, we can
# effectively focus the image along the range axis.
#
# We can synthesize a replica of the Tx pulse from parameters specified in the packet
# metadata. Since we're operating in the frequency domain, we also need to transform our
# pulse replica that we're using as a matched filter to the frequency domain, then take
# the complex conjugate. FInally, we need to multiply every range line by our matched filter.
#
# The Tx pulse replica is given by:
#
# $$\text{Tx Pulse} = exp\biggl\{2i\pi\Bigl(\bigl(\text{TXPSF} + \frac{\text{TXPRR  TXPL}}{2}\bigl)\tau + \frac{\text{TXPRR}}{2}\tau^{2}\Bigl)\biggl\}$$
#
# where $\text{TXPSF}$ is the Tx Pulse Start Frequency, $\text{TXPRR}$ is the Tx Pulse Ramp Rate, and $\text{TXPL}$ is the Tx Pulse Length.


### Create range filter from replica pulse
index_start     = np.ceil((len_range_line - num_tx_vals)/2) - 1;
index_end       = index_start + num_tx_vals - 1; #+ np.ceil((len_range_line - num_tx_vals)/2) - 2;
#
range_filter    = np.zeros(len_range_line, dtype=radar_data.dtype);
range_filter[int(index_start):int(index_end+1)] = tx_replica;
range_filter    = np.conjugate(np.fft.fft(range_filter));

### apply MF filter
#radar_data_mf_f = np.multiply(radar_data_mf_f, range_filter);
np.multiply(radar_data, range_filter, out=radar_data);

### delete stuff which is not more needed
del range_filter, tx_replica, index_start, index_end

#breakpoint()
'''
### Pulse compressed data to be transformed into the Time domain just to display it
radar_data_mf   = np.fft.ifftshift( np.fft.ifft( np.fft.ifft(np.fft.ifftshift(radar_data_f, axes=0), axis=0), axis=1), axes=1);
#radar_data_mf   = np.fft.ifft( np.fft.ifft(radar_data_f, axis=1), axis=0);

radar_data_t   = np.abs(np.fft.ifft(radar_data_f, axis=1)).astype(np.float32);

fig = plt.figure(2, figsize=(8,8), clear=True);
ax  = fig.add_subplot(111);
ax.imshow(np.abs(radar_data_mf), aspect='auto', interpolation='none', );#origin='lower');
ax.set_title(f'{headline} Range compressed', fontweight='bold');
ax.set_xlabel('Fast Time (down range)', fontweight='bold');
ax.set_ylabel('Slow Time (cross range)', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();

#del radar_data_t
'''

"""
### to find some Radar signals in MatchedFilter version

fig = plt.figure(12, figsize=(6, 6), clear=True);
ax  = fig.add_subplot(111);
fs  = range_sample_freq;
#scale = 'linear';
scale = 'dB';
aa, bb, cc, dd = ax.specgram(radar_data_raw[831,:], NFFT=256, Fs=fs/1e6, Fc=None, detrend=None, window=np.hanning(256), scale=scale, noverlap=200, cmap='Greys');
ax.set_xlabel('Time [us]', fontweight='bold');
ax.set_ylabel('Freq [MHz]', fontweight='bold');
ax.set_title('Spectrogram Raw-Data', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();


fig = plt.figure(22, figsize=(6, 6), clear=True);
ax  = fig.add_subplot(111);
aa, bb, cc, dd = ax.specgram(radar_data_mf[831,:], NFFT=256, Fs=fs/1e6, Fc=None, detrend=None, window=np.hanning(256), scale=scale, noverlap=200, cmap='Greys');
ax.set_xlabel('Time [us]', fontweight='bold');
ax.set_ylabel('Freq [MHz]', fontweight='bold');
ax.set_title('Spectrogram MF-Data', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();
"""


#breakpoint()
### 4.4 - Range cell migration correction (RCMC)
# Since the collector motion couples range and azimuth information, point targets will
# tend to produce returns spread in arcs across multiple range bins as the azimuth varies.
# We therefore need to apply a shift to align the phase history associated with each
# pointlike target into a single range bin, so we can then operate 1-dimensionally along
# the azimuth axis to perform azimuth compresison.
#
# The RCMC shift is defined by
#
# $$\text{RCMC shift} = R_{0} \biggl(\frac{1}{D} - 1\biggl)$$
#
# with $D$ being the cosine of the instantaneous squint angle and $R_{0}$ the range of closest approach, both defined in section 4.1. Since we're still operating in the frequency domain we need to apply a filter of the form
#
# $$\text{RCMC filter} = exp\biggl\{4i\pi\frac{f_{\tau}}{c}\bigl(\text{RCMC shift}\bigl)\biggl\}$$
#
# Again, this needs to be multiplied by every range line in our data.

###TODO: eigentlich kommt hier die FFT in Azimuth Richtung

### Create RCMC filter
range_freq_vals = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line)
rcmc_shift      = slant_range_vec[0] * (np.divide(1, D) - 1)
rcmc_filter     = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / c0)

### Apply filter
radar_data      = np.multiply(radar_data, rcmc_filter);

### delete stuff which is not more needed
del rcmc_shift, rcmc_filter, range_freq_vals,
#del radar_data_f


### 4.5 - Convert to range-Doppler domain
# We've finished processing the image in range, so we can inverse FFT back to the range
# domain along the range axis. The image will still be in the frequency domain in azimuth.
radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)


### 4.6 - Azimuth compression - create and apply matched filter
# Our azimuth filter is defined by
# $$\text{Azimuth filter} = exp\biggl\{4j \pi \frac{R_{0} \, D(f_{\eta}, V_{r})}{\lambda}\biggl\}$$

### Create filter
az_filter   = np.exp(4j * np.pi * slant_range_vec * D / wavelength);
### Apply filter
#radar_data  = np.multiply(radar_data, az_filter)
np.multiply(radar_data, az_filter, out=radar_data);
### delete stuff which is not more needed
del az_filter


### 4.7 - Transform back to range-azimuth domain
# Finally, we'll transform back out of the frequency domain by taking the inverse FFT of each azimuth line.
radar_data_final = np.fft.ifft(radar_data, axis=0)


print("STarting FInal Image")
### 5 - Plot Results:   With azimuth compression complete, we're ready to plot our image!
plt.figure(9, figsize=(8,8), clear=True);
#plt.imshow(abs(radar_data[:,:]), vmin=0, vmax=2000, origin='lower')
plt.title(f'{headline} Processed SAR Image', fontweight='bold');
#plt.imshow(abs(radar_data_final), aspect='auto', interpolation='none', origin='lower', norm=colors.LogNorm(vmin=300, vmax=10000))
plt.imshow(20*np.log10(abs(radar_data_final)), aspect='auto', interpolation='none', );#origin='lower')
plt.xlabel('Fast Time (Down Range / samples)', fontweight='bold');
plt.ylabel('Slow Time (Cross Range / samples)', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();


# There are still a few noteworthy issues with our image. The first is folding - notice
# various terrain features from the top of the image are folded into the bottom, and
# terrain from the left of the image is folded into the right side. Folding in range
# (range ambiguities) occurs due to echoes spilling over into earlier or later sampling
# windows. Folding in azimuth occurs due to our sampling the azimuth spectrum of the scene
# at the PRF, which leads to folding in the frequency spectrum.
#
# Various terrain features are clearly visible, however the image is still not perfectly
# focused. We have assumed a Doppler centroid of 0Hz, and have not applied a number of
# additional processing steps that ESA use to produce Level 1 products e.g. Secondary
# Range Compression (SRC). These are left as an exercise for the reader.

