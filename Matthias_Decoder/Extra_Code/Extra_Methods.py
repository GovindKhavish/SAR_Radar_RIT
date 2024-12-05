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
def adaptive_threshold_local(spectrogram_data, block_size):
    """
    Apply local adaptive thresholding to isolate intense signals in a spectrogram.
    
    Parameters:
        spectrogram_data (2D array): The intensity matrix from the spectrogram.
        block_size (int): Size of the local region for thresholding.
        
    Returns:
        2D array: Thresholded spectrogram with values below the local threshold set to zero.
    """
    # Compute the local mean using a uniform filter
    local_mean = uniform_filter(spectrogram_data, size=block_size)
    # Zero out values below the local mean
    thresholded_data = np.where(spectrogram_data > local_mean, spectrogram_data, 0)
    return thresholded_data


# idx_n = 1070
# fs = 46918402.800000004
# radar_section = radar_data[idx_n, :]

# fig = plt.figure(11, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)

# # Generate spectrogram
# aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6, Fc=None, 
#                              detrend=None, window=np.hanning(256), scale='dB', 
#                              noverlap=200, cmap='Greys')

# # Apply local adaptive thresholding
# block_size = 25  # Adjust block size to tune the sensitivity
# thresholded_aa = adaptive_threshold_local(aa, block_size)

# # Re-plot with thresholded data
# fig_thresh = plt.figure(12, figsize=(6, 6), clear=True)
# ax_thresh = fig_thresh.add_subplot(111)
# dd_thresh = ax_thresh.pcolormesh(cc, bb, 10 * np.log10(thresholded_aa + 1e-10), 
#                                  shading='auto', cmap='Greys')  # +1e-10 to avoid log(0)
# cbar_thresh = plt.colorbar(dd_thresh, ax=ax_thresh)
# cbar_thresh.set_label('Intensity [dB]')
# ax_thresh.set_xlabel('Time [us]', fontweight='bold')
# ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
# ax_thresh.set_title(f'Local Adaptive Thresholded Spectrogram from rangeline {idx_n}', fontweight='bold')
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Parameters
idx_n = 1070
fs = 46918402.800000004
NFFT = 256
noverlap = 250
window_size = 10  
threshold_factor = 3  # 3σ threshold

radar_section = radar_data[idx_n, :]

fig, ax = plt.subplots(figsize=(6, 6), clear=True)
Pxx, freqs, bins, im = ax.specgram(radar_section, NFFT=NFFT, Fs=fs / 1e6, detrend=None, window=np.hanning(NFFT), scale='dB', noverlap=noverlap, cmap='Greys')
plt.close(fig) 

Pxx_linear = 10 ** (Pxx / 10)

skewness_map = np.zeros_like(Pxx_linear)

for i in range(Pxx_linear.shape[0] - window_size + 1):
    for j in range(Pxx_linear.shape[1] - window_size + 1):
        window = Pxx_linear[i:i+window_size, j:j+window_size]
        window_flat = window.flatten()
        skewness_map[i, j] = skew(window_flat)

# Compute threshold: 0 + 3σ
sigma1 = np.std(skewness_map)
threshold = 0 + threshold_factor * sigma1

# Identify RFI-flagged bins
rfi_flags = skewness_map > threshold

# Visualize results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original spectrogram
ax[0].imshow(10 * np.log10(Pxx_linear), aspect='auto', cmap='Greys', 
             origin='lower', extent=[bins[0], bins[-1], freqs[0], freqs[-1]])
ax[0].set_title("Original Spectrogram")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Frequency (MHz)")

# Overlay flagged bins
ax[1].imshow(rfi_flags, aspect='auto', cmap='hot', alpha=0.6, 
             origin='lower', extent=[bins[0], bins[-1], freqs[0], freqs[-1]])
ax[1].set_title("RFI-Flagged Bins (Skewness)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Frequency (MHz)")

plt.tight_layout()
plt.show()
