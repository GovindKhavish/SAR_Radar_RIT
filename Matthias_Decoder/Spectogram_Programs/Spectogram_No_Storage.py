#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import polars as pl
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

if _simraddir.exists():
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder

# Mipur VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
# filename = '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Nazareth_Isreal/S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190224t034343-20190224t034416-026066-02e816.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Syria/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Syria/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# filename = '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/NorthernSea_Ireland/S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE"
filename = '/s1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat'


inputfile = filepath + filename

l0file = sentinel1decoder.Level0File(inputfile)

sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

selected_burst = 11
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
global_pulse_number = 1

start_idx = 0
end_idx = 1400
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

    gradient_x = np.gradient(aa_filtered_clean, axis=1)
    gradient_y = np.gradient(aa_filtered_clean, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

   # Define a mask for slash gradients
    slash_mask = (
        ((gradient_x > 0.05) & (gradient_y < -0.05))  # Forward slash `/`
        |  
        ((gradient_x > 0.05) & (gradient_y > 0.05))   # Backslash `\`
    )
    # Apply gradient magnitude threshold
    slash_mask = slash_mask & (gradient_magnitude > np.percentile(gradient_magnitude, 70))

    dilated_mask = binary_dilation(slash_mask, structure=np.ones((6, 6)))

    chirp_candidates = np.where(dilated_mask, aa_filtered_clean, 0)
    min_length = 10
    labeled_mask, num_features = label(dilated_mask)
    slices = find_objects(labeled_mask)
    filtered_mask = np.zeros_like(dilated_mask, dtype=bool)

    for i, slice_obj in enumerate(slices, start=1):
        component = (labeled_mask[slice_obj] == i)
        y_coords, x_coords = np.where(component)
            
        if len(x_coords) > 0 and len(y_coords) > 0:
            length = np.sqrt((x_coords.max() - x_coords.min())**2 + (y_coords.max() - y_coords.min())**2)
                
            if length >= min_length:
                filtered_mask[slice_obj][component] = True

    length_filtered_candidates = np.where(filtered_mask, chirp_candidates, 0)

    # ------------------ Detect Chirp Candidates ------------------
    time_freq_data = np.column_stack(np.where(length_filtered_candidates> 0))
    frequency_indices = bb[time_freq_data[:, 0]]

    if time_freq_data.shape[0] == 0:
        continue  # No data points, skip this range line

    # Apply DBSCAN for clustering
    dbscan = DBSCAN(eps=20, min_samples=5)
    clusters = dbscan.fit_predict(time_freq_data)
    num_clusters = len(np.unique(clusters[clusters != -1]))

    # Skip feature extraction if no valid clusters
    if num_clusters > 2 or num_clusters == 0:
        continue

    # ------------------ Feature Extraction -------------------
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = time_freq_data[clusters == cluster_id]
            frequency_indices = bb[cluster_points[:, 0]]
            time_indices = cluster_points[:, 1]

            bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
            center_frequency = np.mean(frequency_indices)
            time_span = np.max(time_indices) - np.min(time_indices)
            chirp_rate = bandwidth / time_span if time_span != 0 else 0

            start_time = np.min(time_indices) / fs
            end_time = np.max(time_indices) / fs
            adjusted_start_time = start_time + slow_time_offset
            adjusted_end_time = end_time + slow_time_offset
            pulse_duration = adjusted_end_time - adjusted_start_time

            unique_key = (idx_n, cluster_id)

            if unique_key not in global_cluster_params:
                global_cluster_params[unique_key] = []

            global_cluster_params[unique_key].append({
                'pulse_number': global_pulse_number,
                'bandwidth': bandwidth,
                'center_frequency': center_frequency,
                'chirp_rate': chirp_rate,
                'start_time_index': np.min(time_indices),
                'end_time_index': np.max(time_indices),
                'adjusted_start_time': adjusted_start_time,
                'adjusted_end_time': adjusted_end_time,
                'pulse_duration': pulse_duration
            })

            global_pulse_number += 1

        # # ------------------ Process I/Q Data -------------------
        # NFFT = 256
        # noverlap = 200
        # sampling_rate = fs

        # time_step = (NFFT - noverlap) / sampling_rate

        # # Dictionary to store isolated I/Q data for each pulse (global storage)
        # isolated_pulses_data = {}

        # cluster_time_indices = {}
        # for (rangeline_idx, cluster_id), params_list in global_cluster_params.items():
        #     for params in params_list:
        #         start_time_idx = params['start_time_index']
        #         end_time_idx = params['end_time_index']
        #         iq_start_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(start_time_idx, sampling_rate, time_step)
        #         iq_end_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(end_time_idx, sampling_rate, time_step)
                
        #         pulse_number = params['pulse_number']
        #         if pulse_number not in cluster_time_indices:
        #             cluster_time_indices[pulse_number] = []
        #         cluster_time_indices[pulse_number].append((rangeline_idx, cluster_id, iq_start_idx, iq_end_idx))

        # # Initialize isolated data for each pulse
        # for pulse_number in cluster_time_indices:
        #     isolated_pulses_data[pulse_number] = []

        # # Isolate radar data for each pulse
        # for idx in range(len(radar_section)):
        #     for pulse_number, clusters in cluster_time_indices.items():
        #         for (rangeline_idx, cluster_id, iq_start_idx, iq_end_idx) in clusters:
        #             if iq_start_idx <= idx <= iq_end_idx:
        #                 if len(isolated_pulses_data[pulse_number]) <= idx:
        #                     isolated_pulses_data[pulse_number].extend([0] * (idx - len(isolated_pulses_data[pulse_number]) + 1))
        #                 isolated_pulses_data[pulse_number][idx] = radar_section[idx]
        #                 break  # Stop checking once matched

        # # Convert lists to numpy arrays for each pulse
        # for pulse_number in isolated_pulses_data:
        #     isolated_pulses_data[pulse_number] = np.array(isolated_pulses_data[pulse_number], dtype=complex)

        # # Update the global variable with isolated data for this rangeline
        # for pulse_number, data in isolated_pulses_data.items():
        #     if pulse_number not in global_isolated_pulses_data:
        #         global_isolated_pulses_data[pulse_number] = []
        #     global_isolated_pulses_data[pulse_number].append(data)  # Append data from this rangeline

# Plotting
pulse_numbers = []
center_frequencies = []
bandwidths = []
pulse_durations = []
chirp_rates = []

for params_list in global_cluster_params.values():
    for params in params_list:
        pulse_numbers.append(params['pulse_number'])
        center_frequencies.append(params['center_frequency'])
        bandwidths.append(params['bandwidth'])
        pulse_durations.append(params['pulse_duration'])
        chirp_rates.append(params['chirp_rate'])

# Center Frequency vs. Pulse Number
plt.figure(figsize=(10, 6))
plt.plot(pulse_numbers, center_frequencies, linestyle='-', color='b')
plt.xlabel("Pulse Number")
plt.ylabel("Center Frequency (Hz)")
plt.title("Center Frequency vs. Pulse Number")
plt.grid(True)
plt.show()

# Bandwidth vs. Pulse Number
plt.figure(figsize=(10, 6))
plt.plot(pulse_numbers, bandwidths, linestyle='-', color='g')
plt.xlabel("Pulse Number")
plt.ylabel("Bandwidth (Hz)")
plt.title("Bandwidth vs. Pulse Number")
plt.grid(True)
plt.show()

# Pulse Duration vs. Pulse Number
plt.figure(figsize=(10, 6))
plt.plot(pulse_numbers, pulse_durations, linestyle='-', color='r')
plt.xlabel("Pulse Number")
plt.ylabel("Pulse Duration (s)")
plt.title("Pulse Duration vs. Pulse Number")
plt.grid(True)
plt.show()

# Chirp Rate vs. Pulse Number
plt.figure(figsize=(10, 6))
plt.plot(pulse_numbers, chirp_rates, linestyle='-', color='purple')
plt.xlabel("Pulse Number")
plt.ylabel("Chirp Rate (Hz/s)")
plt.title("Chirp Rate vs. Pulse Number")
plt.grid(True)
plt.show()
