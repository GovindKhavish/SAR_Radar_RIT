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
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path, PurePath
#-----------------------------------------------------------------------------------------
# Define the subdirectory path
_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

# Check if the subdirectory exists
if _simraddir.exists():
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder

#-----------------------------------------------------------------------------------------
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'
inputfile = filepath + filename

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

# ------------------ Spectrogram Plot with Local Adaptive Thresholding -------------------
def adaptive_threshold_local(spectrogram_data, threshold_factor):

    mean_val = np.mean(spectrogram_data)
    std_val = np.std(spectrogram_data)
    threshold = mean_val + threshold_factor * std_val
    thresholded_data = np.where(spectrogram_data > threshold, spectrogram_data, 0)
    
    return thresholded_data, threshold

# ------------------ Plotting -------------------
idx_n = 1070
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6, Fc=None, detrend=None, window=np.hanning(256), scale='dB', noverlap=200, cmap='Greys')

# Apply adaptive thresholding 
thresholded_aa,threshold = adaptive_threshold_local(aa, threshold_factor=3)

# Plot
fig_thresh = plt.figure(12, figsize=(6, 6), clear=True)
ax_thresh = fig_thresh.add_subplot(111)
aa_db_filtered = 10 * np.log10(thresholded_aa + 1e-10) 
dd = ax_thresh.imshow(aa_db_filtered, aspect='auto', origin='lower', cmap='Greys')
cbar = plt.colorbar(dd, ax=ax_thresh)
cbar.set_label('Intensity [dB]')
ax_thresh.set_xlabel('Time [us]', fontweight='bold')
ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
ax_thresh.set_title(f'Filtered Spectrogram (Threshold: {round(10 * np.log10(threshold), 2)} dB)', fontweight='bold')
plt.tight_layout()
plt.show()

# DBSCAN 
thresholded_aa_flat = thresholded_aa.flatten()

# 2D array for clustering (time, frequency)
time_freq_data = np.column_stack(np.where(thresholded_aa > 0))  # Get non-zero points

# DBSCAN parameters
dbscan = DBSCAN(eps=3, min_samples=10)
clusters = dbscan.fit_predict(time_freq_data)

# Plot thresholded
fig_thresh = plt.figure(12, figsize=(6, 6), clear=True)
ax_thresh = fig_thresh.add_subplot(111)
aa_db_filtered = 10 * np.log10(thresholded_aa + 1e-10) 
dd = ax_thresh.imshow(aa_db_filtered, aspect='auto', origin='lower', cmap='Greys')

# Plot clusters 
for i, cluster in enumerate(np.unique(clusters)):
    if cluster != -1:  # -1 is noise
        cluster_points = time_freq_data[clusters == cluster]
        ax_thresh.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i}', s=2)

cbar = plt.colorbar(dd, ax=ax_thresh)
cbar.set_label('Intensity [dB]')
ax_thresh.set_xlabel('Time [us]', fontweight='bold')
ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
ax_thresh.set_title(f'Filtered Spectrogram with DBSCAN (Threshold: {round(10 * np.log10(threshold), 2)} dB)', fontweight='bold')
ax_thresh.legend()
plt.tight_layout()


# ------------------ Find Start and End Time Indices for Each Cluster -------------------
# Create a list to store start and end indices for each cluster
cluster_time_indices = {}

# Iterate through each unique cluster (excluding noise cluster -1)
for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude noise points
        # Get the time-frequency points corresponding to the current cluster
        cluster_points = time_freq_data[clusters == cluster_id]
        
        # Get the time indices (the 2nd column of the time_freq_data)
        time_indices = cluster_points[:, 1]  # Column 1 is the time axis (in microseconds)
        
        # Find the start and end time indices for this cluster
        start_time_index = np.min(time_indices)
        end_time_index = np.max(time_indices)
        
        # Store the start and end time indices for the cluster
        cluster_time_indices[cluster_id] = (start_time_index, end_time_index)

# Print out the start and end time indices for each cluster
for cluster_id, (start, end) in cluster_time_indices.items():
    print(f"Cluster {cluster_id}: Start Time Index = {start}, End Time Index = {end}")

plt.show()


