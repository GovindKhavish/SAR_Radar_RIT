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
import Spectogram_FunctionsV2
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
# Mipur VH
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
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

# ------------------ Plotting -------------------
fs = 46918402.800000004


# Define the range of rangelines to process
start_rangeline = 200
end_rangeline = 1500

# Ensure that the range is within the bounds of your radar data
if start_rangeline < 0 or end_rangeline >= len(radar_data):
    raise ValueError(f"Rangeline range {start_rangeline} to {end_rangeline} is out of bounds.")

# Loop through the specified range of rangelines
for idx_n in range(start_rangeline, end_rangeline + 1):
    # Extract the radar section for the current rangeline
    radar_section = radar_data[idx_n, :]
    fig = plt.figure(11, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(111)
    aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale='dB',noverlap=200, cmap='Greys')

    # Apply adaptive threshold
    threshold, aa_db_filtered = Spectogram_FunctionsV2.adaptive_threshold_local(radar_section, 2)

    # DBSCAN clustering
    eps=3 
    min_samples = 10
    clusters, frequency_indices, time_freq_data = Spectogram_FunctionsV2.apply_dbscan(aa_db_filtered, bb, eps, min_samples)

    # Extract cluster parameters
    cluster_time_indices, cluster_params = Spectogram_FunctionsV2.extract_cluster_params(time_freq_data, clusters, bb)

    # Print out the parameters for each cluster (optional)
    for cluster_id, params in cluster_params.items():
        print(f"Cluster {cluster_id}:")
        print(f"  Bandwidth: {params['bandwidth']} MHz")
        print(f"  Center Frequency: {params['center_frequency']} MHz")
        print(f"  Chirp Rate: {params['chirp_rate']} MHz/us")
        print(f"  Start Time Index: {params['start_time_index']}")
        print(f"  End Time Index: {params['end_time_index']}")
        print('---------------------------')

    # Isolate the radar data based on the detected clusters (same as original logic)

    # Define the spectrogram parameters
    NFFT = 256
    noverlap = 200
    sampling_rate = fs  # Sampling rate of the I/Q data

    time_step = (NFFT - noverlap) / sampling_rate  #  (seconds)

    mapped_cluster_indices = {}
    for cluster_id, params in cluster_params.items():
        start_time_idx = params['start_time_index']
        end_time_idx = params['end_time_index']
        iq_start_idx = Spectogram_FunctionsV2.spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
        iq_end_idx = Spectogram_FunctionsV2.spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
        mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

    # Initialize isolated radar data
    isolated_radar_data = np.zeros_like(radar_section, dtype=complex)

    # Isolate radar data based on the cluster indices
    for idx in range(len(radar_section)):
        for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
            if iq_start_idx <= idx <= iq_end_idx:
                isolated_radar_data[idx] = radar_section[idx]
                break  # Stop checking once matched
