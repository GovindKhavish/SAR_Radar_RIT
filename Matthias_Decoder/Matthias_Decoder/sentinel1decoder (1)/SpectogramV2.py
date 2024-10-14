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

# # Plotting the result
# plt.figure(figsize=(14, 6))
# plt.imshow(10*np.log10(abs(radar_data[:,:])), aspect='auto', interpolation='none', origin='lower') #vmin=0,vmax=10)
# plt.colorbar(label='Amplitude')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.title('Orginal Data')


# -------------------- Coherent Compression -----------------------------#
# Parameters
idx_n = 1500
fs = 46918402.800000004

radar_section = radar_data[idx_n,:]#[:,idx_n]

fig = plt.figure(10, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
ax.plot(np.abs(radar_section), label=f'abs{idx_n}')
ax.plot(np.real(radar_section), label=f'Re{idx_n}')
ax.plot(np.imag(radar_section), label=f'Im{idx_n}')
ax.legend()
ax.set_title(f'{headline} Raw I/Q Sensor Output', fontweight='bold')
ax.set_xlabel('Fast Time (down range) [samples]', fontweight='bold')
ax.set_ylabel('|Amplitude|', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# # Define filter cutoff frequencies (positive counterparts of -1 MHz to -8 MHz)
# lowcut = 1e6   # 1 MHz in Hz
# highcut = 8e6  # 8 MHz in Hz

# # Design a bandpass filter to isolate the chirp
# b, a = butter(N=4, Wn=[lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')

# # Apply the filter to the IQ data
# radar_section_filtered = filtfilt(b, a, radar_section)

# # Compute and plot the spectrogram after filtering
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)

# ax.specgram(radar_section_filtered, NFFT=512, Fs=fs/1e6, window=np.hanning(512), scale='dB', noverlap=450, cmap='Greys')

# ax.set_xlabel('Time [us]', fontweight='bold')
# ax.set_ylabel('Frequency [MHz]', fontweight='bold')
# ax.set_title('Spectrogram After Bandpass Filtering (1 MHz to 8 MHz)', fontweight='bold')

# plt.tight_layout()
# plt.show()

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)

scale = 'dB'
aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,
noverlap=200, cmap='Greys')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# Compute the magnitude of the IQ data for rangeline 1500
magnitude = np.abs(radar_section)

# Define a threshold to detect the chirp (adjust based on your data)
threshold = np.max(magnitude) * 0.5  # 50% of the maximum value (adjust if necessary)

# Find the fast time indices where the chirp occurs (where the magnitude exceeds the threshold)
chirp_indices = np.where(magnitude > threshold)[0]

# Isolate the chirp by extracting the fast time section corresponding to the chirp
y = 200
start_idx = np.min(chirp_indices)
end_idx = np.max(chirp_indices)
isolated_chirp = radar_section[start_idx-y:end_idx+y]

# Visualize the IQ data (real, imaginary, and magnitude) for the isolated chirp
plt.figure(figsize=(10, 6))
plt.plot(np.abs(isolated_chirp), label='|Amplitude|')
plt.plot(np.real(isolated_chirp), label='Real Part')
plt.plot(np.imag(isolated_chirp), label='Imaginary Part')
plt.legend()
plt.title(f'Isolated Chirp in Rangeline {idx_n}')
plt.xlabel('Fast Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)


# Compute and plot the spectrogram of the isolated chirp
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
Pxx, freqs, bins, im = ax.specgram(isolated_chirp, NFFT=256, Fs=fs/1e6, window=np.hanning(256), scale='dB', noverlap=200, cmap='Greys')

# Customize the axes
ax.set_xlabel('Time [μs]', fontweight='bold')
ax.set_ylabel('Frequency [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram of Isolated Chirp in Rangeline {idx_n}', fontweight='bold')

import mne
from mne.time_frequency import psd_array_multitaper

# Calculate power spectral density using multitaper with magnitude
isolated_chirp_magnitude = np.abs(isolated_chirp)
psd, freqs = psd_array_multitaper(isolated_chirp_magnitude, sfreq=fs, fmin=0, fmax=fs/2, adaptive=True, normalization='full', verbose=0)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(freqs, psd)
plt.title('Multitaper Spectral Estimation (Magnitude)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Power')
plt.grid(True)
plt.show()


# Show the plot
plt.tight_layout()
plt.show()

# # -------------------- Power Density -----------------------------#
# from scipy.signal import welch, periodogram

# # Parameters for Welch's method
# nperseg = 256
# noverlap = 200

# # Assuming radar_data and idx_n are defined
# # frequencies, psd= periodogram(np.real(radar_data[idx_n, :]), fs=fs)

# frequencies, psd = welch(np.real(radar_section), fs=fs, nperseg=nperseg, noverlap=noverlap)

# # Get maximum PSD value in linear scale
# max_psd_value_linear = np.max(psd)

# # Convert to dB
# max_psd_value_db = 10 * np.log10(max_psd_value_linear)

# # Get the frequency corresponding to the maximum PSD value
# max_psd_freq = frequencies[np.argmax(psd)]

# # Print maximum PSD value in dB and corresponding frequency
# print(f"Maximum PSD Value (dB): {max_psd_value_db}")
# print(f"Corresponding Frequency: {max_psd_freq} Hz")

# fig = plt.figure(12, figsize=(6, 6), clear=True)
# ax = fig.add_subplot(111)
# ax.plot(frequencies / 1e6, 10 * np.log10(psd), label=f'PSD (Welch) - Line {idx_n}')
# ax.set_xlabel('Frequency [MHz]', fontweight='bold')
# ax.set_ylabel('Power/Frequency [dB/Hz]', fontweight='bold')
# ax.set_title(f'Power Spectral Density (PSD) from rangeline {idx_n}', fontweight='bold')
# ax.legend()
# plt.tight_layout()
# plt.show()

# -------------------- Threshold Function -----------------------------#
def threshold_filter_and_plot(data, threshold_db, fs, title):

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
# lowcut = 0.1  # Lower bound (MHz)
# highcut = 10  # Upper bound (MHz)
# nyquist = fs / 2e6  # Nyquist frequency in MHz
# low = lowcut / nyquist
# high = highcut / nyquist

# b, a = butter(4, [low, high], btype='band')
# filtered_radar_data = filtfilt(b, a, radar_section)

# # print("\nBandpass Filtering Statistics:")
# # print(f'Maximum: {np.max(filtered_radar_data)}')
# # print(f'Minimum: {np.min(filtered_radar_data)}')
# # print(f'Mean: {np.mean(filtered_radar_data)}')

# # Plot the filtered data spectrogram
# fig, ax = plt.subplots(figsize=(6, 6), clear=True)
# aa, bb, cc, dd = ax.specgram(filtered_radar_data, 
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
# ax.set_title(f'Spectrogram of Filtered Data (0-10 MHz)', fontweight='bold')
# ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
# plt.tight_layout()


# -------------------- Windowing Technique -----------------------------#
# hann_window = windows.hann(len(radar_section))
# windowed_data = radar_section * hann_window

# # print("\nWindowing Technique Statistics:")
# # print(f'Maximum: {np.max(windowed_data)}')
# # print(f'Minimum: {np.min(windowed_data)}')
# # print(f'Mean: {np.mean(windowed_data)}')

# # Plot the windowed data spectrogram
# fig, ax = plt.subplots(figsize=(6, 6), clear=True)
# aa, bb, cc, dd = ax.specgram(windowed_data, 
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
# ax.set_title(f'Spectrogram of Windowed Data (Hanning) 0-10 MHz', fontweight='bold')
# ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
# plt.tight_layout()


# -------------------- FFT Masking -----------------------------#
# fft_radar_data = np.fft.fft(radar_section)
# freqs = np.fft.fftfreq(len(fft_radar_data), d=1/fs)

# mask = (np.abs(freqs) > 10e6)
# fft_radar_data[mask] = 0 
# filtered_radar_data_ifft = np.fft.ifft(fft_radar_data)

# # print("\nFFT Masking Statistics:")
# # print(f'Maximum: {np.max(filtered_radar_data_ifft)}')
# # print(f'Minimum: {np.min(filtered_radar_data_ifft)}')
# # print(f'Mean: {np.mean(filtered_radar_data_ifft)}')

# # Plot the IFFT filtered data spectrogram
# fig, ax = plt.subplots(figsize=(6, 6), clear=True)
# aa, bb, cc, dd = ax.specgram(filtered_radar_data_ifft, 
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
# ax.set_title(f'Frequency Domain Filtered Spectrogram (0-10 MHz)', fontweight='bold')
# ax.set_ylim([-10, 10])  # Focus on 0-10 MHz
# plt.tight_layout()

plt.show()

#threshold_value = max_psd_value_db  # Set your threshold value here

# # For Bandpass Filtered Data
# print("\nApplying Threshold Filtering for Bandpass Filtered Data:")
# bandpass_filtered_data_with_threshold = threshold_filter_and_plot(filtered_radar_data, threshold_value, fs, "Bandpass Filtered Data Spectrogram")

# # For Windowed Data
# print("\nApplying Threshold Filtering for Windowed Data:")
# windowed_data_with_threshold = threshold_filter_and_plot(windowed_data, threshold_value, fs, "Windowed Data Spectrogram")

# # For FFT Masked Data
# print("\nApplying Threshold Filtering for FFT Masked Data:")
# fft_masked_data_with_threshold = threshold_filter_and_plot(filtered_radar_data_ifft, threshold_value, fs, "FFT Masked Data Spectrogram")