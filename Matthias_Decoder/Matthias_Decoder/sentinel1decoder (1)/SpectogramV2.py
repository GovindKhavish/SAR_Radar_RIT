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

# Load the Level 0 File
l0file = sentinel1decoder.Level0File(inputfile)

# Extract Metadata and Burst Information
sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

# Select the Burst to Process
selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

headline = f'Sentinel-1 (burst {selected_burst}): '

# Extract Raw I/Q Sensor Data
radar_data = l0file.get_burst_data(selected_burst)


# -------------------- Coherent Compression -----------------------------#
# Parameters
idx_n = 1500;
fs = 46918402.800000004;

fig = plt.figure(10, figsize=(6, 6), clear=True);
ax = fig.add_subplot(111);
ax.plot(np.abs(radar_data[idx_n,:]), label=f'abs{idx_n}');
ax.plot(np.real(radar_data[idx_n,:]), label=f'Re{idx_n}');
ax.plot(np.imag(radar_data[idx_n,:]), label=f'Im{idx_n}');
ax.legend();
ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold');
ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold');
ax.set_ylabel('|Amplitude|', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();


fig = plt.figure(11, figsize=(6, 6), clear=True);
ax = fig.add_subplot(111);

#scale = 'linear';
scale = 'dB';
aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,
Fc=None, detrend=None, window=np.hanning(256), scale=scale,
noverlap=200, cmap='Greys');
ax.set_xlabel('Time [us]', fontweight='bold');
ax.set_ylabel('Freq [MHz]', fontweight='bold');
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold');
plt.tight_layout(); plt.pause(0.1); plt.show();

# -------------------- Threshold Function -----------------------------#
import numpy as np
import matplotlib.pyplot as plt

def threshold_filter_and_plot(data, threshold_db, fs, title):
    """
    Apply a threshold to the input data, zeroing out values below the threshold,
    and plot the resulting data as a spectrogram.

    Parameters:
        data (np.ndarray): Input data to apply threshold.
        threshold_db (float): Threshold value in dB.
        fs (float): Sampling frequency in Hz.
        title (str): Title for the spectrogram plot.
    """
    # Convert threshold from dB to linear scale for comparison
    threshold_linear = 10 ** (threshold_db / 10)

    # Zero out values below the threshold
    filtered_data = np.where(np.abs(data) < threshold_linear, 0, data)

    # Plot the modified data as a spectrogram
    fig, ax = plt.subplots(figsize=(6, 6), clear=True)
    _, _, _, im = ax.specgram(filtered_data, 
                               NFFT=256, 
                               Fs=fs / 1e6,  # Convert to MHz for plotting
                               Fc=None, 
                               detrend=None, 
                               window=np.hanning(256), 
                               scale='dB', 
                               noverlap=200, 
                               cmap='Greys')

    ax.set_xlabel('Time [μs]', fontweight='bold')
    ax.set_ylabel('Freq [MHz]', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
    plt.colorbar(im, ax=ax, label='Intensity [dB]')  # Use the correct object for colorbar
    plt.tight_layout()
    plt.show()
    
    return filtered_data



# # -------------------- Chirp Matched -----------------------------#
# # Chirp Matched Filtering
# from scipy.signal import chirp, correlate

# # Generate a reference chirp (you'll need the correct chirp parameters)
# t = np.linspace(0, 1, len(radar_data[idx_n, :]), endpoint=False)
# chirp_signal = chirp(t, f0=0, f1=10e6, t1=1, method='linear')  # Chirp 0-10 MHz

# # Perform matched filtering (cross-correlation with the chirp)
# matched_filter_output = correlate(radar_data[idx_n, :], chirp_signal, mode='same')

# # Calculate and print statistics for matched filter output
# print("\nChirp Matched Filtering Statistics:")
# print(f'Maximum: {np.max(matched_filter_output)}')
# print(f'Minimum: {np.min(matched_filter_output)}')
# print(f'Mean: {np.mean(matched_filter_output)}')

# # Plot the matched filter output spectrogram
# fig, ax = plt.subplots(figsize=(6, 6), clear=True)
# aa, bb, cc, dd = ax.specgram(matched_filter_output, 
#                              NFFT=256, 
#                              Fs=fs / 1e6, 
#                              Fc=None, 
#                              detrend=None, 
#                              window=np.hanning(256), 
#                              scale='dB', 
#                              noverlap=200, 
#                              cmap='Greys')

# ax.set_xlabel('Time [μs]', fontweight='bold')
# ax.set_ylabel('Freq [MHz]', fontweight='bold')
# ax.set_title(f'Matched Filter Output Spectrogram (0-10 MHz)', fontweight='bold')
# ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
# plt.tight_layout()


# -------------------- Bandpass Filtering -----------------------------#
lowcut = 0.1  # Lower bound (MHz)
highcut = 10  # Upper bound (MHz)
nyquist = fs / 2e6  # Nyquist frequency in MHz
low = lowcut / nyquist
high = highcut / nyquist

b, a = butter(4, [low, high], btype='band')
filtered_radar_data = filtfilt(b, a, radar_data[idx_n, :])

# print("\nBandpass Filtering Statistics:")
# print(f'Maximum: {np.max(filtered_radar_data)}')
# print(f'Minimum: {np.min(filtered_radar_data)}')
# print(f'Mean: {np.mean(filtered_radar_data)}')

# Plot the filtered data spectrogram
fig, ax = plt.subplots(figsize=(6, 6), clear=True)
aa, bb, cc, dd = ax.specgram(filtered_radar_data, 
                             NFFT=256, 
                             Fs=fs / 1e6, 
                             Fc=None, 
                             detrend=None, 
                             window=np.hanning(256), 
                             scale='dB', 
                             noverlap=200, 
                             cmap='Greys')

ax.set_xlabel('Time [μs]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram of Filtered Data (0-10 MHz)', fontweight='bold')
ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
plt.tight_layout()


# -------------------- Windowing Technique -----------------------------#
hann_window = windows.hann(len(radar_data[idx_n, :]))
windowed_data = radar_data[idx_n, :] * hann_window

# print("\nWindowing Technique Statistics:")
# print(f'Maximum: {np.max(windowed_data)}')
# print(f'Minimum: {np.min(windowed_data)}')
# print(f'Mean: {np.mean(windowed_data)}')

# Plot the windowed data spectrogram
fig, ax = plt.subplots(figsize=(6, 6), clear=True)
aa, bb, cc, dd = ax.specgram(windowed_data, 
                             NFFT=256, 
                             Fs=fs / 1e6, 
                             Fc=None, 
                             detrend=None, 
                             window=np.hanning(256), 
                             scale='dB', 
                             noverlap=200, 
                             cmap='Greys')

ax.set_xlabel('Time [μs]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram of Windowed Data (Hanning) 0-10 MHz', fontweight='bold')
ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
plt.tight_layout()


# -------------------- FFT Masking -----------------------------#
fft_radar_data = np.fft.fft(radar_data[idx_n, :])
freqs = np.fft.fftfreq(len(fft_radar_data), d=1/fs)

mask = (np.abs(freqs) > 10e6)
fft_radar_data[mask] = 0 
filtered_radar_data_ifft = np.fft.ifft(fft_radar_data)

# print("\nFFT Masking Statistics:")
# print(f'Maximum: {np.max(filtered_radar_data_ifft)}')
# print(f'Minimum: {np.min(filtered_radar_data_ifft)}')
# print(f'Mean: {np.mean(filtered_radar_data_ifft)}')

# Plot the IFFT filtered data spectrogram
fig, ax = plt.subplots(figsize=(6, 6), clear=True)
aa, bb, cc, dd = ax.specgram(filtered_radar_data_ifft, 
                             NFFT=256, 
                             Fs=fs / 1e6, 
                             Fc=None, 
                             detrend=None, 
                             window=np.hanning(256), 
                             scale='dB', 
                             noverlap=200, 
                             cmap='Greys')

ax.set_xlabel('Time [μs]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Frequency Domain Filtered Spectrogram (0-10 MHz)', fontweight='bold')
ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
plt.tight_layout()

plt.show()

threshold_value = 10  # Set your threshold value here

# For Bandpass Filtered Data
print("\nApplying Threshold Filtering for Bandpass Filtered Data:")
bandpass_filtered_data_with_threshold = threshold_filter_and_plot(filtered_radar_data, threshold_value, fs, "Bandpass Filtered Data Spectrogram")

# For Windowed Data
print("\nApplying Threshold Filtering for Windowed Data:")
windowed_data_with_threshold = threshold_filter_and_plot(windowed_data, threshold_value, fs, "Windowed Data Spectrogram")

# For FFT Masked Data
print("\nApplying Threshold Filtering for FFT Masked Data:")
fft_masked_data_with_threshold = threshold_filter_and_plot(filtered_radar_data_ifft, threshold_value, fs, "FFT Masked Data Spectrogram")