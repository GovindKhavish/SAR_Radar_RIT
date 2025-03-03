#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import polars as pl
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from sklearn.linear_model import RANSACRegressor
from skimage.morphology import binary_dilation
import math

#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path
#-----------------------------------------------------------------------------------------

_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

if _simraddir.exists():
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder

# Mipur VH Filepath
#filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Nazareth_Isreal/S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190224t034343-20190224t034416-026066-02e816.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Syria/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Syria/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/NorthernSea_Ireland/S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE"
# filename = '/s1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat'


inputfile = filepath + filename

l0file = sentinel1decoder.Level0File(inputfile)

sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

while selection['Signal Type'].unique()[0] != 0:
    selected_burst += 1
    selection = l0file.get_burst_metadata(selected_burst)

headline = f'Sentinel-1 (burst {selected_burst}): '

radar_data = l0file.get_burst_data(selected_burst)

plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(abs(radar_data[:, :])), aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Original Data')
plt.show()

#------------------------ Intial Spectogram --------------------------------
idx_n = 560
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

plt.figure(figsize=(14, 6))
plt.plot(np.abs(radar_section), color='b', linewidth=1.5)
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Magnitude', fontweight='bold')
plt.title(f'Absolute Value of IQ Data for Row {idx_n}', fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'
aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')

ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.show()

#------------------------ Chirp Formulation--------------------------------
fs = 46918402.8  # Hz
bw = 5e6  # Hz
fc = -4e6  # Hz
chirp_duration_us = 10  # microseconds
chirp_duration_s = chirp_duration_us * 1e-6  # seconds
row_idx = 560  # Row index
chirp_rate_fixed_bw = (bw / chirp_duration_s)  # Chirp rate Hz/s

bw_mhz = 3  # MHz
chirp_rate_mhz_per_us = bw_mhz / chirp_duration_us
print(f"Chirp Rate: {chirp_rate_mhz_per_us:.2f} MHz/us")

# Generate chirp signal
num_samples = int((chirp_duration_s) * fs) 
T = num_samples / fs  
t = np.linspace(0, T, num_samples)
start_freq = fc - bw / 2  
end_freq = fc + bw / 2  

chirp_signal = 50 * np.exp(1j * (2 * np.pi * (start_freq * t + 0.5 * chirp_rate_fixed_bw * t**2)))

num_cols = radar_data.shape[1]
start_idx = num_cols // 2  
end_idx = start_idx + num_samples  

if end_idx > num_cols:
    raise ValueError("Chirp duration is too long for the row length.")

radar_data_original = radar_data[row_idx, :].copy()

# Padded
chirp_signal_padded = np.zeros(num_cols, dtype=complex)
chirp_signal_padded[start_idx:end_idx] = chirp_signal
radar_data[row_idx, :] += chirp_signal_padded 

plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(abs(radar_data[row_idx, :]) + np.finfo(float).eps), 'r', linewidth=1.5, label='Modified')
plt.plot(10 * np.log10(abs(radar_data_original) + np.finfo(float).eps), 'k', linewidth=1.5, label='Original')

plt.axvline(start_idx, linestyle='--', color='b', linewidth=1.5, label="Start")
plt.axvline(end_idx, linestyle='--', color='b', linewidth=1.5, label="End")

plt.xlabel('Fast Time Index', fontweight='bold')
plt.ylabel('Magnitude (dB)', fontweight='bold')
plt.title(f'Overlay of Row {row_idx} Before and After Chirp Injection', fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------- Spectrogram  -------------------
fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'
NFFT = 256
noverlap = 200
window = np.hanning(NFFT)

aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {row_idx}', fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
def adaptive_threshold(array, factor=1):
    mean_value = np.mean(array)
    std_value = np.std(array)
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

threshold,aa = adaptive_threshold(aa)

plt.figure(figsize=(10, 5))
plt.imshow(np.flipud(aa), interpolation='none', aspect='auto', extent=[cc[-1], cc[0], bb[0], bb[-1]]) 
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()
plt.show()

# Radar data dimensions
time_size = aa.shape[1] # Freq
freq_size = aa.shape[0] # Time

# Create 2D Mask
#vert_guard,vert_avg,hori_Guard,hori_avg
vert_guard = 12
vert_avg = 30
hori_guard = 10
hori_avg = 30
alarm_rate = 1e-9

cfar_mask = Spectogram_FunctionsV3.create_2d_mask(vert_guard,vert_avg,hori_guard,hori_avg)

padded_mask = Spectogram_FunctionsV3.create_2d_padded_mask(aa,cfar_mask)

alpha = Spectogram_FunctionsV3.set_alpha(Spectogram_FunctionsV3.get_total_average_cells(vert_guard,vert_avg,hori_guard,hori_avg),alarm_rate)

thres_map = Spectogram_FunctionsV3.cfar_method(aa,padded_mask,alpha)

aa_db_filtered = Spectogram_FunctionsV3.detect_targets(aa, thres_map)

plt.figure(figsize=(10, 5))
plt.imshow(np.flipud(aa_db_filtered), interpolation='none', aspect='auto', extent=[cc[-1], cc[0], bb[0], bb[-1]])
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()

# Assume aa_filtered_clean is the spectrogram in dB format
aa_filtered_clean = aa_db_filtered  # Use your existing spectrogram data
# Create a filtered radar data array where values correspond to the non-zero entries of the CFAR mask
filtered_radar_data = aa * aa_filtered_clean

# Create a new array to store the filtered spectrogram data (keep values where CFAR mask is non-zero)
filtered_spectrogram_data = np.zeros_like(aa)  # Initialize with zeros (same shape as aa)
filtered_spectrogram_data[aa_filtered_clean > 0] = aa[aa_filtered_clean > 0]

# # Visualize the filtered spectrogram
# plt.figure(figsize=(10, 5))
# plt.imshow(filtered_spectrogram_data, cmap='jet', origin='lower', aspect='auto')
# plt.title("Filtered Spectrogram (Only Extracted Values)")
# plt.colorbar(label="Intensity")
# plt.xlabel("Time (samples)")
# plt.ylabel("Frequency (Hz)")
# plt.tight_layout()
# plt.show()

# Apply binary dilation to widen the detected shapes (slashes)
dilated_mask = binary_dilation(aa_filtered_clean, footprint=np.ones((1, 1)))

# Label the connected components in the dilated binary mask
labeled_mask, num_labels = label(dilated_mask, connectivity=2, return_num=True)

# Define thresholds
min_angle = 5
max_angle = 85
min_diagonal_length = 10
min_aspect_ratio = 1

# Create empty mask for valid slashes
filtered_mask_slashes = np.zeros_like(dilated_mask, dtype=bool)

# Debug visualization
plt.figure(figsize=(10, 5))
plt.imshow(dilated_mask, cmap='gray', origin='lower', aspect='auto')
plt.title("Detected Regions and Filtered Slashes")
plt.xlabel("Time (samples)")
plt.ylabel("Frequency (Hz)")

for region in regionprops(labeled_mask):
    minr, minc, maxr, maxc = region.bbox
    diagonal_length = np.hypot(maxr - minr, maxc - minc)

    # Skip small regions
    if diagonal_length < min_diagonal_length:
        continue

    # Compute width, height, and aspect ratio
    width = maxc - minc
    height = maxr - minr
    aspect_ratio = max(width, height) / (min(width, height) + 1e-5)

    # Ensure elongated shape
    if aspect_ratio < min_aspect_ratio:
        continue

    # Compute slope and angle
    slope = height / width if width != 0 else float('inf')
    angle = np.degrees(np.arctan(slope))
    angle = abs(angle)

    is_forward_slash = min_angle <= angle <= max_angle
    is_backward_slash = (180 - max_angle) <= angle <= (180 - min_angle)

    if not (is_forward_slash or is_backward_slash):
        continue

    # Extract pixel coordinates of the region
    coords = np.array(region.coords)
    y_vals, x_vals = coords[:, 0], coords[:, 1]

    # Fit a RANSAC regression model
    ransac = RANSACRegressor()
    ransac.fit(x_vals.reshape(-1, 1), y_vals)  # Fit the model

    # Get the R² score (how well the line fits)
    r2_score = ransac.score(x_vals.reshape(-1, 1), y_vals)

    # Set a lower R² threshold to allow slight variations
    min_r2_threshold = 0.85

    if r2_score < min_r2_threshold:
        continue  # Skip non-straight shapes


    # If passed all checks, add to final mask
    filtered_mask_slashes[labeled_mask == region.label] = True

    # Debug: Draw bounding box
    plt.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], 'r-', linewidth=1)

plt.tight_layout()
plt.show()

# Display final filtered mask
plt.figure(figsize=(10, 5))
plt.imshow(filtered_mask_slashes, cmap='gray', origin='lower', aspect='auto')
plt.title("Final Filtered Mask (Only Straight Slashes)")
plt.xlabel("Time (samples)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
# Extract non-zero points as time-frequency data for clustering but are the indices from the spectogram
time_freq_data = np.column_stack(np.where(filtered_mask_slashes > 0))
# DBSCAN Clustering
clusters = DBSCAN(eps=20, min_samples=1).fit_predict(time_freq_data)
# Map the frequency indices to MHz
frequencies_mhz = bb[time_freq_data[:, 0]]  # Convert frequency indices to MHz
# Map the time indices to microseconds
time_us = cc[time_freq_data[:, 1]]  # Time indices in µs
plt.figure(figsize=(10, 5))
plt.scatter(time_us, frequencies_mhz, c=clusters, cmap='viridis', s=5)
plt.title('DBSCAN Clustering of Chirp Signals')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Cluster ID')
plt.tight_layout()
plt.show()

# Plot the target map (filtered spectrogram)
plt.figure(figsize=(10, 5))
plt.imshow(np.flipud(aa_db_filtered), interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[0], bb[-1]])
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')

for i, cluster in enumerate(np.unique(clusters[clusters != -1])):  # Exclude noise points
    cluster_points = time_freq_data[clusters == cluster]
    cluster_time_us = cc[cluster_points[:, 1]]  # Time in µs
    cluster_freq_mhz = bb[cluster_points[:, 0]]  # Frequency in MHz
    plt.scatter(cluster_time_us, cluster_freq_mhz, c='r', label=f'Cluster {i}', s=5, edgecolors='none', marker='o')

plt.tight_layout()
plt.legend()
plt.show(block=True)
plt.show()

# Number of clusters (excluding noise)
num_clusters = len(np.unique(clusters[clusters != -1]))
print(f"Number of clusters: {num_clusters}")

# ------------------ Start and End Time for Each Cluster -------------------
cluster_time_indices = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Noise

        # Time-freq points for cluster
        cluster_points = time_freq_data[clusters == cluster_id]
        # Time indices (2nd column)
        time_indices = cc[cluster_points[:, 1]]  # Time axis (us)
        
        # Start and End time
        start_time_index = np.min(time_indices)
        end_time_index = np.max(time_indices)

        cluster_time_indices[cluster_id] = (start_time_index, end_time_index)

# Extract Cluster Parameters
cluster_params = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude noise 
        cluster_points = time_freq_data[clusters == cluster_id]
        frequency_indices = bb[cluster_points[:, 0]]  # Use the correct frequency bins (bb)
        
        # 2nd column of the time_freq_data
        time_indices = cluster_points[:, 1]  # us
        
        bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
        center_frequency = (np.max(frequency_indices) + np.min(frequency_indices)) / 2
        time_span = np.max(time_indices) - np.min(time_indices)  # us
        if time_span != 0:
            chirp_rate = bandwidth / time_span  # MHz per us
        else:
            chirp_rate = 0 
        
        cluster_params[cluster_id] = {
            'bandwidth': bandwidth,
            'center_frequency': center_frequency,
            'chirp_rate': chirp_rate,
            'start_time_index': np.min(time_indices),
            'end_time_index': np.max(time_indices)
        }

for cluster_id, params in cluster_params.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Bandwidth: {params['bandwidth']} MHz")
    print(f"  Center Frequency: {params['center_frequency']} MHz")
    print(f"  Chirp Rate: {params['chirp_rate']} MHz/us")
    print(f"  Start Time Index (us): {params['start_time_index']}")
    print(f"  End Time Index (us): {params['end_time_index']}")
    print('---------------------------')

# # Numpy Array conversition
# cluster_params_array = np.array([[cluster_id, params['bandwidth'], params['center_frequency'], params['chirp_rate'], params['start_time_index'], params['end_time_index']]
#                                  for cluster_id, params in cluster_params.items()])

NFFT = 256
noverlap = 200
sampling_rate = fs
time_step = (NFFT - noverlap) / sampling_rate  # seconds

# Map cluster indices to I/Q start and end indices
mapped_cluster_indices = {}
for cluster_id, params in cluster_params.items():
    start_time_idx = params['start_time_index']
    end_time_idx = params['end_time_index']
    iq_start_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
    iq_end_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
    mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

# Initialize a dictionary to store isolated radar data for each cluster
isolated_pulses_data = {}

# Populate the isolated I/Q data for each cluster
for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
    isolated_pulses_data[cluster_id] = np.zeros_like(radar_section, dtype=complex)  # Zero-initialized array
    for idx in range(len(radar_section)):
        if iq_start_idx <= idx <= iq_end_idx:  # Check if index is within the cluster range
            isolated_pulses_data[cluster_id][idx] = radar_section[idx]

# Check if there are any isolated pulses data (i.e., clusters)
if len(isolated_pulses_data) > 0:
    # Visualize the isolated data for each cluster
    fig, axes = plt.subplots(len(isolated_pulses_data), 1, figsize=(10, 6), sharex=True, sharey=True)

    # If there's only one cluster, make sure axes is not a list
    if len(isolated_pulses_data) == 1:
        axes = [axes]

    # Plot each cluster's isolated I/Q data
    for idx, (cluster_id, iq_data) in enumerate(isolated_pulses_data.items()):
        # Plot the I/Q data (real and imaginary parts)
        axes[idx].plot(np.real(iq_data), label=f"Cluster {cluster_id} - Real", color='blue')
        axes[idx].plot(np.imag(iq_data), label=f"Cluster {cluster_id} - Imaginary", color='red')
        
        axes[idx].set_title(f"Cluster {cluster_id} - Isolated I/Q Data")
        axes[idx].set_xlabel("Index")
        axes[idx].set_ylabel("Amplitude")
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

else:
    print("No clusters detected, skipping the isolated I/Q data visualization.")


