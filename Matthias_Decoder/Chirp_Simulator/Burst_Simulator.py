#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import polars as pl
import pandas as pd
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from sklearn.linear_model import RANSACRegressor
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
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
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

selected_burst = 17
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
# Define parameters for the 5 chirp signals
bw_list = [5e6, 6e6, 7e6, 8e6]  # Hz
fc_list = [-4e6, -2e6, 2e6, 4e6]  # Hz
chirp_durations = [10, 15, 20, 25]  # microseconds
fs = 46918402.8  # Sampling frequency in Hz

# Generate chirp signals
chirp_signals = []
num_samples_list = []
for bw, fc, chirp_duration_us in zip(bw_list, fc_list, chirp_durations):
    chirp_duration_s = chirp_duration_us * 1e-6  # Convert to seconds
    chirp_rate = bw / chirp_duration_s  # Chirp rate (Hz/s)
    num_samples = int(chirp_duration_s * fs)
    num_samples_list.append(num_samples)
    
    t = np.linspace(0, chirp_duration_s, num_samples)
    start_freq = fc - bw / 2
    end_freq = fc + bw / 2
    
    chirp_signal = 50 * np.exp(1j * (2 * np.pi * (start_freq * t + 0.5 * chirp_rate * t**2)))
    chirp_signals.append(chirp_signal)

# Inject chirps into every third row at random positions
num_rows, num_cols = radar_data.shape
chirp_count = {i: 0 for i in range(4)}  # Keep count of injected chirps

for row_idx in range(0, num_rows, 3):  # Every third row
    chirp_idx = np.random.randint(0, 4)  # Randomly select a chirp
    chirp_signal = chirp_signals[chirp_idx]
    num_samples = num_samples_list[chirp_idx]
    
    start_idx = np.random.randint(0, num_cols - num_samples)  # Random start index
    end_idx = start_idx + num_samples
    
    radar_data[row_idx, start_idx:end_idx] += chirp_signal  # Inject chirp
    chirp_count[chirp_idx] += 1  # Track injected chirp count

# Print chirp injection statistics
for i, count in chirp_count.items():
    print(f"Chirp {i+1} (BW={bw_list[i]/1e6}MHz, FC={fc_list[i]/1e6}MHz, Duration={chirp_durations[i]}us) injected {count} times")

plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(abs(radar_data[:, :])), aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Original Data')
plt.show()

#------------------------ Apply CFAR filtering --------------------------------
global_pulse_number = 1
start_idx = 0
end_idx = radar_data.shape[0] - 1 
fs = 46918402.800000004  

global_cluster_params = {}
global_isolated_pulses_data = {}

for idx_n in range(start_idx, end_idx + 1):
    radar_section = radar_data[idx_n, :]
    slow_time_offset = idx_n / fs 

    # ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
    fig = plt.figure(11, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(111)
    scale = 'dB'
    aa, bb, cc, dd = ax.specgram(
        radar_data[idx_n, :], 
        NFFT=256, 
        Fs=fs / 1e6, 
        detrend=None, 
        window=np.hanning(256), 
        scale=scale, 
        noverlap=200, 
        cmap='Greys'
    )

    # -------------------- Adaptive Threshold on Intensity Data -----------------------------#
    threshold,aa = Spectogram_FunctionsV3.adaptive_threshold(aa)

    #------------------------ Apply CFAR filtering --------------------------------
    # Radar data dimensions
    #radar_data = radar_data[0:200,10000:15000]
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

    # ------------------- Shape Detection ---------------------
    aa_filtered_clean = aa_db_filtered  
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
    min_angle = 15
    max_angle = 80
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

    # ------------------ Detect Chirp Candidates ------------------
    time_freq_data = np.column_stack(np.where(filtered_mask_slashes > 0))
    # DBSCAN Clustering
    if time_freq_data.shape[0] > 0:
        clusters = DBSCAN(eps=20, min_samples=1).fit_predict(time_freq_data)
        print(idx_n)
    else:
        print("No data points to cluster.")

    # Map the frequency indices to MHz
    frequencies_mhz = bb[time_freq_data[:, 0]]  # Convert frequency indices to MHz
    # Map the time indices to microseconds
    time_us = cc[time_freq_data[:, 1]]  # Time indices in µs

    if time_freq_data.shape[0] == 0:
        continue  # No data points, skip this range line

    num_clusters = len(np.unique(clusters[clusters != -1]))

    # Skip feature extraction if no valid clusters
    if num_clusters > 2 or num_clusters == 0:
        continue

    # ------------------ Feature Extraction -------------------
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = time_freq_data[clusters == cluster_id]
            frequency_indices = bb[cluster_points[:, 0]]
            iq_indices = cluster_points[:, 1]
            time_indices = cc[cluster_points[:, 1]]

            bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
            center_frequency = np.mean(frequency_indices)
            time_span = np.max(time_indices) - np.min(time_indices)
            chirp_rate = bandwidth / time_span if time_span != 0 else 0

            start_time = np.min(time_indices) / fs
            end_time = np.max(time_indices) / fs
            adjusted_start_time = start_time + slow_time_offset
            adjusted_end_time = end_time + slow_time_offset
            pulse_duration = adjusted_end_time - adjusted_start_time

            start_iq = np.min(iq_indices) 
            end_iq = np.max(iq_indices)  

            unique_key = (idx_n, cluster_id)

            if unique_key not in global_cluster_params:
                global_cluster_params[unique_key] = []

            print('--')
            print(bandwidth)
            print(center_frequency)
            print(pulse_duration)

            global_cluster_params[unique_key].append({
                'pulse_number': global_pulse_number,
                'bandwidth': bandwidth,
                'center_frequency': center_frequency,
                'chirp_rate': chirp_rate,
                'start_time_index': np.min(time_indices),
                'end_time_index': np.max(time_indices),
                'adjusted_start_time': adjusted_start_time,
                'adjusted_end_time': adjusted_end_time,
                'pulse_duration': pulse_duration,
                'start_time_iq': np.min(start_iq),
                'end_time_iq': np.max(end_iq)
            })

            global_pulse_number += 1


def bin_values(values, tolerance):
    """Bins values based on a % threshold, checking against all existing bins."""
    values = np.sort(values)
    bins = [values[0]]  # Start with the first value
    
    for val in values[1:]:
        # Check if value fits into any existing bin
        fits_existing_bin = any(abs(val - b) / b <= tolerance for b in bins)
        
        if not fits_existing_bin:  # Create a new bin only if it doesn't fit anywhere
            bins.append(val)
    
    return np.digitize(values, bins, right=True)


def pdw_analysis(df, tolerance):
    """Performs PDW analysis on the provided DataFrame."""
    # Apply binning to group center frequency and chirp rate based on tolerance
    df["center_freq_bin"] = bin_values(df["center_frequency"].values, tolerance)
    df["chirp_rate_bin"] = bin_values(df["chirp_rate"].values, tolerance)

    # Group by these bins and calculate required statistics
    grouped_df = df.groupby(["center_freq_bin", "chirp_rate_bin"]).agg(
        pulse_count=("pulse_number", "count"),
        mean_pulse_duration=("pulse_duration", "mean"),
        min_pulse_duration=("pulse_duration", "min"),
        max_pulse_duration=("pulse_duration", "max"),
        mean_center_frequency=("center_frequency", "mean"),  # Added mean center frequency
        mean_chirp_rate=("chirp_rate", "mean")               # Added mean chirp rate
    ).reset_index()

    # Return the grouped DataFrame
    return grouped_df


def plot_pdw_bins(df, tolerance):
    """Plot rectangular bins for Center Frequency vs. Chirp Rate."""
    
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Define color mapping for unique bins
    unique_bins = df["center_freq_bin"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    color_map = dict(zip(unique_bins, colors))

    for _, row in df.iterrows():
        # Compute bin ranges with 5% tolerance
        freq_min = row["mean_center_frequency"] * (1 - tolerance)
        freq_max = row["mean_center_frequency"] * (1 + tolerance)
        chirp_min = row["mean_chirp_rate"] * (1 - tolerance)
        chirp_max = row["mean_chirp_rate"] * (1 + tolerance)
        
        # Create a rectangle
        rect = patches.Rectangle(
            (freq_min, chirp_min),  # Bottom-left corner
            freq_max - freq_min,    # Width (frequency range)
            chirp_max - chirp_min,  # Height (chirp rate range)
            linewidth=1.5,
            edgecolor=color_map[row["center_freq_bin"]],
            facecolor="none",
            linestyle="--",
            label=f'Bin {row["center_freq_bin"]}' if row["center_freq_bin"] not in plt.gca().get_legend_handles_labels()[1] else ""
        )
        ax.add_patch(rect)

    # Labels and title
    plt.xlabel("Center Frequency (Hz)")
    plt.ylabel("Chirp Rate (Hz/s)")
    plt.title("Binned Center Frequency vs Chirp Rate (with Tolerance)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Center Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.xlim(df["mean_center_frequency"].min() * 0.95, df["mean_center_frequency"].max() * 1.05)
    plt.ylim(df["mean_chirp_rate"].min() * 0.95, df["mean_chirp_rate"].max() * 1.05)
    plt.show()


def plot_pdw_scatter(df):
    """Scatter plot of Center Frequency vs. Chirp Rate using the actual values."""

    # Ensure correct column names
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count", "mean_center_frequency", "mean_chirp_rate"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    plt.figure(figsize=(8, 6))

    # Define color mapping for unique frequency bins
    unique_bins = df["center_freq_bin"].unique()
    color_map = plt.cm.get_cmap("tab10", len(unique_bins))  # Using a predefined colormap

    # Scatter plot with actual center frequency and chirp rate
    for i, bin_value in enumerate(unique_bins):
        # Filter data by bin
        bin_data = df[df["center_freq_bin"] == bin_value]
        # Plot data for each bin with a unique color
        plt.scatter(
            bin_data["mean_center_frequency"] / 1e6,  # Convert center frequency to MHz
            bin_data["mean_chirp_rate"]/ 1e12,        # Convert chirp rate to MHz/us
            s=bin_data["pulse_count"] * 5,            # Scale marker size by pulse count
            color=color_map(i),                       # Use the color from colormap
            alpha=0.6
        )

    # Labels and title
    plt.xlabel("Center Frequency (MHz)")
    plt.ylabel("Chirp Rate (MHz/us)")
    plt.title("Center Frequency vs Chirp Rate (Grouped by Tolerance)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# Convert global_cluster_params to a DataFrame
all_params = []

for unique_key, params in global_cluster_params.items():
    for param in params:
        all_params.append(param)

df = pd.DataFrame(all_params)

# Ensure that the DataFrame has the necessary columns
df["pulse_number"] = df["pulse_number"]
df["center_frequency"] = df["center_frequency"]
df["chirp_rate"] = df["chirp_rate"]
df["pulse_duration"] = df["pulse_duration"]

# Perform PDW analysis with your desired tolerance value
tolerance = 0.05  # Set tolerance as needed
grouped_df = pdw_analysis(df, tolerance)

# Now you can plot the results
plot_pdw_bins(grouped_df, tolerance)
plot_pdw_scatter(grouped_df)

