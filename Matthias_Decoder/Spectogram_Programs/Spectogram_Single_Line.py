#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import binary_dilation, label, find_objects
from sklearn.cluster import DBSCAN
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path
#-----------------------------------------------------------------------------------------
_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

# Check if the subdirectory exists
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

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Syria/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

# filepath = r"//Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Nazareth_Isreal/S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190224t034343-20190224t034416-026066-02e816.dat'

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

#------------------------ Apply CFAR filtering --------------------------------
# Spectrogram plot
idx_n = 1470
fs = 46918402.800000004
radar_section = radar_data[idx_n, :]

fig = plt.figure(11, figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'
aa, bb, cc, dd = ax.specgram(radar_data[idx_n,:], NFFT=256, Fs=fs/1e6,Fc=None, detrend=None, window=np.hanning(256), scale=scale,noverlap=200, cmap='Greys')

ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.pause(0.1)

# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
def adaptive_threshold(array, factor=2):
    mean_value = np.mean(array)
    std_value = np.std(array)
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
    
    return threshold,thresholded_array

threshold,aa = adaptive_threshold(aa)

fig = plt.figure(figsize=(6, 6), clear=True)
ax = fig.add_subplot(111)
dd = ax.imshow(10 * np.log10(aa), aspect='auto', origin='lower', cmap='Greys')
cbar = plt.colorbar(dd, ax=ax)
cbar.set_label('Intensity [dB]')
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Filtered Spectrogram (Threshold: {round(10*np.log10(threshold),2)} dB)', fontweight='bold')
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

# # Plot the CFAR Mask
# plt.figure(figsize=(2, 10))
# plt.imshow(cfar_mask, interpolation='none', aspect='auto')
# plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.colorbar(label='Filter Amplitude')

padded_mask = Spectogram_FunctionsV3.create_2d_padded_mask(aa,cfar_mask)

# Plot the Padded Mask
# plt.figure(figsize=(2, 10))
# plt.imshow(padded_mask, interpolation='none', aspect='auto')
# plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
# plt.xlabel('Fast Time')
# plt.ylabel('Slow Time')
# plt.colorbar(label='Filter Amplitude')

alpha = Spectogram_FunctionsV3.set_alpha(Spectogram_FunctionsV3.get_total_average_cells(vert_guard,vert_avg,hori_guard,hori_avg),alarm_rate)

# thres_map = cfar_method(aa,cfar_mask,alpha)
thres_map = Spectogram_FunctionsV3.cfar_method(aa,padded_mask,alpha)

# Plot the Threshold Map
# plt.figure(figsize=(10, 5))
# plt.imshow(thres_map, interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[0], bb[-1]])
# plt.title('Threshold map')
# plt.xlabel('Time [us]')
# plt.ylabel('Frequency [MHz]')
# plt.colorbar(label='Filter Amplitude')
# plt.tight_layout()

# Detect the targets using the spectrogram data
aa_db_filtered = Spectogram_FunctionsV3.detect_targets(aa, thres_map)

# Plot the Target Map
plt.figure(figsize=(10, 5))
plt.imshow(np.flipud(aa_db_filtered), interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[-1], bb[0]])
plt.title('Targets')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Filter Amplitude')
plt.tight_layout()

# Assume aa_filtered_clean is the spectrogram in dB format
aa_filtered_clean = aa_db_filtered  # Use your existing spectrogram data

# Compute gradients to highlight linear patterns in the spectrogram
gradient_x = np.gradient(aa_filtered_clean, axis=1)  # Time (horizontal axis)
gradient_y = np.gradient(aa_filtered_clean, axis=0)  # Frequency (vertical axis)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gradient X")
plt.imshow(gradient_x, cmap='coolwarm')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Gradient Y")
plt.imshow(gradient_y, cmap='coolwarm')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap='hot')
plt.colorbar()

plt.tight_layout()
plt.show()

# Define a mask forslash gradients
slash_mask = (
    ((gradient_x > 0.05) & (gradient_y < -0.05))  # Forward slash `/`
    |  
    ((gradient_x > 0.05) & (gradient_y > 0.05))   # Backslash `\`
)
# Apply gradient magnitude threshold
slash_mask = slash_mask & (gradient_magnitude > np.percentile(gradient_magnitude, 70))

# Convert mask to float for better visualization (optional)
visual_mask = slash_mask.astype(float)

plt.figure(figsize=(6, 6))
plt.imshow(visual_mask, cmap="gray", interpolation="nearest")
plt.title("Slash Mask (Forward `/` and Backslash `\\`)")
plt.axis("off")  # Hide axis labels for a clean look
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(slash_mask.astype(float), cmap="gray", interpolation="nearest")
plt.title("Slash Mask (Forward `/` and Backslash `\\`)")
plt.axis("off")
plt.show()

    
# Plot the original forward_slash_mask
plt.figure(figsize=(12, 5))

# Plot original forward slash mask
plt.subplot(1, 2, 1)
plt.imshow(slash_mask, cmap='gray', origin='lower')  # origin='lower' to align it properly
plt.title("Original Forward Slash Mask")
plt.colorbar(label="Mask Value (True/False)")

# Apply binary dilation and plot the result
dilated_mask = binary_dilation(slash_mask, structure=np.ones((6, 6)))

# Plot dilated mask
plt.subplot(1, 2, 2)
plt.imshow(dilated_mask, cmap='gray', origin='lower')  # origin='lower' to align it properly
plt.title("Dilated Forward Slash Mask")
plt.colorbar(label="Mask Value (True/False)")

plt.show()
# Step 1: Plot the original spectrogram
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the original spectrogram (before mask is applied)
ax[0].imshow(aa_filtered_clean, aspect='auto', cmap='gray', origin='lower')
ax[0].set_title("Original Spectrogram (Before Mask)")
ax[0].set_xlabel("Time (samples)")
ax[0].set_ylabel("Frequency (Hz)")

# Step 2: Apply the mask to the spectrogram and plot
chirp_candidates = np.where(dilated_mask, aa_filtered_clean, 0)

# Plot the chirp candidates (after applying the mask)
ax[1].imshow(chirp_candidates, aspect='auto', cmap='jet', origin='lower')
ax[1].set_title("Spectrogram with Chirp Candidates")
ax[1].set_xlabel("Time (samples)")
ax[1].set_ylabel("Frequency (Hz)")

# Show the plots
plt.tight_layout()
plt.show()

# Threshold for minimum line length (in pixels, accounting for angles)
min_length = 10

# Label connected components in the dilated mask
labeled_mask, num_features = label(dilated_mask)

# Find slices for each labeled component
slices = find_objects(labeled_mask)

# Create a new mask to include only components with sufficient length
filtered_mask = np.zeros_like(dilated_mask, dtype=bool)

for i, slice_obj in enumerate(slices, start=1):
    component = (labeled_mask[slice_obj] == i)
    y_coords, x_coords = np.where(component)
    
    # Measure the bounding box diagonal (approximates major axis length)
    if len(x_coords) > 0 and len(y_coords) > 0:
        length = np.sqrt((x_coords.max() - x_coords.min())**2 + (y_coords.max() - y_coords.min())**2)
        
        # Include the component if its length is greater than the threshold
        if length >= min_length:
            filtered_mask[slice_obj][component] = True

# Apply the filtered mask to extract candidates that meet the length criterion
length_filtered_candidates = np.where(filtered_mask, chirp_candidates, 0)

import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(filtered_mask.astype(float), cmap="gray", interpolation="nearest")
ax.set_title("Filtered Mask with Bounding Boxes")

# Plot bounding boxes and labels
for i, slice_obj in enumerate(slices, start=1):
    y_start, y_end = slice_obj[0].start, slice_obj[0].stop
    x_start, x_end = slice_obj[1].start, slice_obj[1].stop

    # Draw rectangle
    rect = patches.Rectangle(
        (x_start, y_start),  # Bottom-left corner
        x_end - x_start,     # Width
        y_end - y_start,     # Height
        linewidth=1.5, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # Add label at top-left corner of the bounding box
    ax.text(x_start, y_start - 2, str(i), color="yellow", fontsize=8, fontweight="bold")

plt.axis("off")  # Hide axis labels
plt.show()

# Visualize the chirp candidates after applying gradient and dilation
plt.figure(figsize=(10, 5))
plt.imshow(10 * np.log10(chirp_candidates + 1e-10),interpolation='none',aspect='auto',cmap='Greys',origin='lower')
plt.title('Filtered Chirp Candidates ')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()

# Visualize the chirp candidates after length filtering
plt.figure(figsize=(10, 5))
plt.imshow(10 * np.log10(length_filtered_candidates + 1e-10),interpolation='none',aspect='auto',cmap='Greys',origin='lower')
plt.title('Filtered Chirp Candidates')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------

# DBSCAN Clustering
# Extract non-zero points as time-frequency data for clustering
time_freq_data = np.column_stack(np.where(length_filtered_candidates > 0))

# Perform DBSCAN clustering
clusters = DBSCAN(eps=4, min_samples=10).fit_predict(time_freq_data)

# Visualize Clustering Results
plt.figure(figsize=(10, 5))
plt.scatter(time_freq_data[:, 1], time_freq_data[:, 0], c=clusters, cmap='viridis', s=5)
plt.title('DBSCAN Clustering of Chirp Signals')
plt.xlabel('Time [us]')
plt.ylabel('Frequency [MHz]')
plt.colorbar(label='Cluster ID')
plt.tight_layout()
plt.show()

#---------

# Plot threshold
fig_thresh = plt.figure(13, figsize=(6, 6), clear=True)
ax_thresh = fig_thresh.add_subplot(111)
aa_db_filtered = 10 * np.log10(aa_db_filtered  + 1e-10) 
dd = ax_thresh.imshow(aa_db_filtered, aspect='auto', origin='lower', cmap='Greys')

for i, cluster in enumerate(np.unique(clusters[clusters != -1])):  # Exclude noise here
    cluster_points = time_freq_data[clusters == cluster]
    ax_thresh.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i}', s=2)

cbar = plt.colorbar(dd, ax=ax_thresh)
cbar.set_label('Intensity [dB]')
ax_thresh.set_xlabel('Time [us]', fontweight='bold')
ax_thresh.set_ylabel('Freq [MHz]', fontweight='bold')
ax_thresh.set_title(f'Filtered Spectrogram with DBSCAN (Threshold: dB)', fontweight='bold')
ax_thresh.legend()
plt.tight_layout()


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
        time_indices = cluster_points[:, 1]  # Time axis (us)
        
        # Start and End time
        start_time_index = np.min(time_indices)
        end_time_index = np.max(time_indices)

        cluster_time_indices[cluster_id] = (start_time_index, end_time_index)

for cluster_id, (start, end) in cluster_time_indices.items():
    print(f"Cluster {cluster_id}: Start Time Index = {start}, End Time Index = {end}")

# Extract Cluster Parameters
cluster_params = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude noise 
        cluster_points = time_freq_data[clusters == cluster_id]
        frequency_indices = bb[cluster_points[:, 0]]  # Use the correct frequency bins (bb)
        
        # 2nd column of the time_freq_data
        time_indices = cluster_points[:, 1]  # us
        

        bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
        center_frequency = np.mean(frequency_indices)
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
    print(f"  Start Time Index: {params['start_time_index']}")
    print(f"  End Time Index: {params['end_time_index']}")
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
