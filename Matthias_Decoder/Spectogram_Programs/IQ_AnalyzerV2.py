#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import pandas as pd
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.signal import welch
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path
#-----------------------------------------------------------------------------------------
# Define the subdirectory path

from skimage import filters
print(filters.threshold_otsu)

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
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE/"
filename = 's1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

# Damascus VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Damascus_Syria\S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE"
# # # filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE/"
# filename = '\s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat'

# Whitesands VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\WhiteSand_USA\2021_12_14\S1A_IW_RAW__0SDV_20211214T130351_20211214T130423_041005_04DEF2_011D.SAFE"
# filename = '\s1a-iw-raw-s-vh-20211214t130351-20211214t130423-041005-04def2.dat'

# Israel VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Nazareth_Isreal\S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE"
# filename = '\s1a-iw-raw-s-vh-20190224t034343-20190224t034416-026066-02e816.dat'

# Northern Sea Ireland VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\NorthernSea_Ireland\S1A_IW_RAW__0SDV_20200705T181540_20200705T181612_033323_03DC5B_2E3A.SAFE"
# filename = '\s1a-iw-raw-s-vh-20200705t181540-20200705t181612-033323-03dc5b.dat'

# Augsberg VH Filepath
# filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Augsburg_Germany\S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE"
# filename = '\s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat'

inputfile = filepath + filename

l0file = sentinel1decoder.Level0File(inputfile)

sent1_meta = l0file.packet_metadata
bust_info = l0file.burst_info
sent1_ephe = l0file.ephemeris

global_pulse_number = 1
global_cluster_params = {}
global_isolated_pulses_data = {}

echo_bursts = l0file.burst_info[l0file.burst_info['Signal Type'] == 0]
burst_array = np.array(echo_bursts['Burst']) # type: ignore
print("Echo Burst Numbers:", burst_array)


#for selected_burst in burst_array:
selected_burst = 57
selection = l0file.get_burst_metadata(selected_burst)

print(f"Processing Burst {selected_burst}...")
radar_data = l0file.get_burst_data(selected_burst)

#------------------------ Apply CFAR filtering --------------------------------
p = 1350
start_idx = p
end_idx = p #radar_data.shape[0] - 1 
fs = 46918402.800000004  

for idx_n in range(start_idx, end_idx + 1):
    radar_section = radar_data[idx_n, :]
    slow_time_offset = idx_n / fs 
    plt.figure(figsize=(14, 6))
    plt.plot(np.abs(radar_section), color='b', linewidth=1.5)
    plt.xlabel('Sample Index', fontweight='bold')
    plt.ylabel('Magnitude', fontweight='bold')
    plt.title(f'Absolute Value of IQ Data for Row {idx_n}', fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
    fig = plt.figure(11, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(111)
    scale = 'dB'
    aa, bb, cc, dd = ax.specgram(radar_data[idx_n, :], NFFT=256, Fs=fs / 1e6, detrend=None, window=np.hanning(256), scale=scale, noverlap=200, cmap='Greys')
    # bb is in MHz and aa is in us as fs/1e6
    # -------------------- Adaptive Threshold on Intensity Data -----------------------------#
    threshold,aa = Spectogram_FunctionsV3.adaptive_threshold(aa)

    #------------------------ Apply CFAR filtering --------------------------------
    # Radar data dimensions
    time_size = aa.shape[1] # Freq
    freq_size = aa.shape[0] # Time

    # Create 2D Mask
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
    filtered_radar_data = aa * aa_filtered_clean
    filtered_spectrogram_data = np.zeros_like(aa) 
    filtered_spectrogram_data[aa_filtered_clean > 0] = aa[aa_filtered_clean > 0]

    labeled_mask, num_labels = label(aa_filtered_clean, connectivity=2, return_num=True)

    min_angle = 10
    max_angle = 85
    min_diagonal_length = 15
    min_aspect_ratio = 1

    filtered_mask_slashes = np.zeros_like(aa_filtered_clean, dtype=bool)

    # Main loop
    for region in regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        diagonal_length = np.hypot(maxr - minr, maxc - minc)

        if diagonal_length < min_diagonal_length:
            continue

        width = maxc - minc
        height = maxr - minr
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)

        if aspect_ratio < min_aspect_ratio:
            continue

        slope = height / width if width != 0 else float('inf')
        angle = np.degrees(np.arctan(slope))
        angle = abs(angle)

        is_forward_slash = min_angle <= angle <= max_angle
        is_backward_slash = (180 - max_angle) <= angle <= (180 - min_angle)

        if not (is_forward_slash or is_backward_slash):
            continue

        coords = np.array(region.coords)
        y_vals, x_vals = coords[:, 0], coords[:, 1]

        ransac = RANSACRegressor()
        ransac.fit(x_vals.reshape(-1, 1), y_vals)

        # R^2 
        try:
            r2_score = ransac.score(x_vals.reshape(-1, 1), y_vals)
        except ValueError:
            continue  # R^2 calculation fails

        # Slight variations
        min_r2_threshold = 0.85

        if r2_score < min_r2_threshold:
            continue  

        filtered_mask_slashes[labeled_mask == region.label] = True

    # ---------------------------------------------------------
    time_freq_data = np.column_stack(np.where(filtered_mask_slashes > 0))

    # DBSCAN
    if time_freq_data.shape[0] == 0:
        continue  # No targets detected
    else: 
        dbscan = DBSCAN(eps=20, min_samples=5)
        clusters = dbscan.fit_predict(time_freq_data)
        num_clusters = len(np.unique(clusters[clusters != -1]))

        # ------------------ Skip Feature Extraction if More Than 2 Clusters -------------------
        if (num_clusters > 2 or num_clusters == 0):
            continue

        # ------------------ Assign Global Pulse Numbers and Adjusted Times -------------------
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  
                cluster_points = time_freq_data[clusters == cluster_id]
                frequency_indices = bb[cluster_points[:, 0]] # Freq in Mhz

                time_indices = cc[cluster_points[:, 1]] # Time in us
                start_time = np.min(time_indices)
                end_time = np.max(time_indices)

                bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
                center_frequency = (np.max(frequency_indices) + np.min(frequency_indices)) / 2
                time_span = start_time - end_time
                chirp_rate = bandwidth / time_span if time_span != 0 else 0

                adjusted_start_time = start_time + slow_time_offset
                adjusted_end_time = end_time + slow_time_offset
                pulse_duration = adjusted_end_time - adjusted_start_time

                unique_key = global_pulse_number  

                if unique_key not in global_cluster_params:
                    global_cluster_params[unique_key] = []

                global_cluster_params[unique_key].append({
                    'burst_number': selected_burst,
                    'rangeline_number': idx_n,
                    'pulse_number': global_pulse_number,
                    'cluster_id': cluster_id,
                    'bandwidth': bandwidth,
                    'center_frequency': center_frequency,
                    'chirp_rate': chirp_rate,
                    'start_time': start_time,
                    'end_time': end_time,
                    'adjusted_start_time': adjusted_start_time,
                    'adjusted_end_time': adjusted_end_time,
                    'pulse_duration': pulse_duration
                })
                # Print cluster info right here
                print(f"Cluster ID: {cluster_id} | Bandwidth: {bandwidth:.3f} | Center Frequency: {center_frequency:.3f}")
                print(f"Start Time: {np.min(time_indices)} | End Time : {np.max(time_indices)}")
                global_pulse_number += 1

def estimate_bandwidth_1(iq_data, fs, max_components=10, percentile_thresh=5, coverage=(0.01, 0.99)):

    nperseg = min(len(iq_data), 1024)
    freqs, psd = welch(iq_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, return_onesided=False)
    psd = psd / np.max(psd)
    psd_smooth = gaussian_filter1d(psd, sigma=0.1)

    threshold = np.percentile(psd_smooth, percentile_thresh)
    mask = psd_smooth > threshold
    freqs_clean = freqs[mask]
    psd_clean = psd_smooth[mask]

    replication_factor = 1000
    scaled_weights = (psd_clean * replication_factor).astype(int)
    replicated_freqs = np.repeat(freqs_clean, scaled_weights)
    X = replicated_freqs.reshape(-1, 1)

    lowest_bic = np.inf
    best_gmm = None
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    pdf_vals = np.exp(best_gmm.score_samples(freqs.reshape(-1, 1)))
    pdf_vals /= np.max(pdf_vals)  # Normalize
    cdf = np.cumsum(pdf_vals)
    cdf /= cdf[-1]

    lower_idx = np.argmax(cdf >= coverage[0])
    upper_idx = np.argmax(cdf >= coverage[1])

    lower = freqs[lower_idx]
    upper = freqs[upper_idx]
    bandwidth_mhz = (upper - lower) / 1e6

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e6, psd_smooth, label="Smoothed PSD", color='black')
    plt.plot(freqs / 1e6, pdf_vals, '--', label="GMM PDF", color='red')
    plt.axvline(lower / 1e6, color='blue', linestyle=':', label=f"{int(coverage[0]*100)}% Threshold")
    plt.axvline(upper / 1e6, color='green', linestyle=':', label=f"{int(coverage[1]*100)}% Threshold")
    plt.title("Bandwidth Estimation via GMM PDF")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"[PDF Method] Estimated Bandwidth ≈ {bandwidth_mhz:.3f} MHz")
    return bandwidth_mhz


def estimate_bandwidth(iq_data, fs, max_components=10, coverage=(0.01, 0.99)):

    nperseg = min(len(iq_data), 1024)
    freqs, psd = welch(iq_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, return_onesided=False, scaling='density')
    psd = np.fft.fftshift(psd)
    freqs = np.fft.fftshift(freqs)

    psd = psd / np.max(psd)
    psd_smooth = gaussian_filter1d(psd, sigma=0.1)

    freqs_clean = freqs
    psd_clean = psd_smooth

    replication_factor = 1000
    scaled_weights = (psd_clean * replication_factor).astype(int)
    replicated_freqs = np.repeat(freqs_clean, scaled_weights)
    X = replicated_freqs.reshape(-1, 1)

    lowest_bic = np.inf
    best_gmm = None
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    # PDF and CDF
    freqs_sorted_idx = np.argsort(freqs)
    freqs_sorted = freqs[freqs_sorted_idx]
    pdf_vals = np.exp(best_gmm.score_samples(freqs_sorted.reshape(-1, 1)))
    pdf_vals /= np.max(pdf_vals)
    cdf = np.cumsum(pdf_vals)
    cdf /= cdf[-1]

    lower_idx = np.argmax(cdf >= coverage[0])
    upper_idx = np.argmax(cdf >= coverage[1])

    lower = freqs_sorted[lower_idx]
    upper = freqs_sorted[upper_idx]
    bandwidth_mhz = (upper - lower) / 1e6

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e6, psd_smooth, label="Smoothed PSD", color='black')
    plt.plot(freqs_sorted / 1e6, pdf_vals, '--', label="GMM PDF", color='red')
    plt.axvline(lower / 1e6, color='blue', linestyle=':', label=f"{int(coverage[0]*100)}% Threshold")
    plt.axvline(upper / 1e6, color='green', linestyle=':', label=f"{int(coverage[1]*100)}% Threshold")
    plt.title("Bandwidth Estimation via GMM PDF")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"[GMM PDF Method] Estimated Bandwidth ≈ {bandwidth_mhz:.3f} MHz")

    return bandwidth_mhz


NFFT = 256
noverlap = 200
sampling_rate = fs
time_step = (NFFT - noverlap) / sampling_rate  # seconds

mapped_pulse_indices = {}

for global_pulse_number, param_list in global_cluster_params.items():
    params = param_list[0]
    start_time_idx = params['start_time']
    end_time_idx = params['end_time']
    
    iq_start_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(start_time_idx, sampling_rate)
    iq_end_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(end_time_idx, sampling_rate)

    mapped_pulse_indices[global_pulse_number] = (iq_start_idx, iq_end_idx)

isolated_pulses_data = {}
bandwidth_results = {}

for pulse_num, (iq_start_idx, iq_end_idx) in mapped_pulse_indices.items():
    extension = 4 * (iq_end_idx - iq_start_idx)
    start_idx = max(0, iq_start_idx - extension)
    end_idx = min(len(radar_section), iq_end_idx + extension + 1)
    pure_signal = radar_section[start_idx:end_idx]

    signal_mag = np.abs(pure_signal)
    smoothed_mag = uniform_filter1d(signal_mag, size=5)

    # Log-compress the signal magnitude
    compressed_mag = np.log1p(smoothed_mag)

    # Compute Otsu threshold on compressed data
    otsu_thresh = threshold_otsu(compressed_mag)

    # Detection mask on compressed magnitude
    detection_map = (compressed_mag > otsu_thresh).astype(int)

    isolated_pulses_data[pulse_num] = {
        'iq_data': pure_signal,
        'detection_mask': detection_map,
        'threshold_map': np.full_like(signal_mag, np.expm1(otsu_thresh)),  # back-transform for visualization
        'cfar_filtered_signal': pure_signal[detection_map == 1]
    }

    plt.figure(figsize=(10, 4))
    plt.plot(signal_mag, label="|IQ Signal|", color='blue', alpha=0.7)
    plt.plot(np.full_like(signal_mag, np.expm1(otsu_thresh)), label="Log-Comp Otsu Threshold", color='orange', linestyle='--')
    detected_indices = np.where(detection_map == 1)[0]
    plt.scatter(detected_indices, signal_mag[detected_indices], color='green', label='Detections', s=30)
    plt.title(f"Pulse {pulse_num} - Otsu Thresholding")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


    filtered_signal = isolated_pulses_data[pulse_num]['cfar_filtered_signal']
    if len(filtered_signal) > 0:
        #bandwidth_mhz = estimate_bandwidth_1(filtered_signal, fs=sampling_rate)
        bandwidth_mhz = estimate_bandwidth(filtered_signal, fs=sampling_rate)
        # or
        #bandwidth = estimate_bandwidth(filtered_signal, fs=sampling_rate, method='percentile')
    else:
        bandwidth_mhz = 0.0  # Fallback if no detections

    # Store result per pulse
    bandwidth_results[pulse_num] = bandwidth_mhz

    # (Optional) Add to per-pulse dictionary for traceability
    isolated_pulses_data[pulse_num]['bandwidth_mhz'] = bandwidth_mhz

# # ---------------- CFAR Parameters ---------------- #
# for pulse_num, (iq_start_idx, iq_end_idx) in mapped_pulse_indices.items():
#     isolated_legnth = (iq_end_idx - iq_start_idx)
#     total_cells = iq_end_idx - iq_start_idx + 1
#     guard_cells =  10
#     training_cells = 10


#     cfar_mask = Spectogram_FunctionsV3.create_1d_mask(guard_cells, training_cells)

#     plt.figure(figsize=(8, 3))
#     plt.stem(cfar_mask, basefmt=" ")
#     plt.title('1D CFAR Mask')
#     plt.xlabel('Cell Index')
#     plt.ylabel('Mask Value')
#     plt.grid(True)
#     plt.show()

#     extension = 2 * isolated_legnth
#     start_idx = max(0, iq_start_idx - extension)
#     end_idx = min(len(radar_section), iq_end_idx + extension + 1)
#     pure_signal = radar_section[start_idx:end_idx]

#     # ---------- CFAR 1D Filtering ---------- #
#     alpha = Spectogram_FunctionsV3.set_alpha((2 * training_cells), alarm_rate)
#     threshold_map = Spectogram_FunctionsV3.cfar_method_1d(pure_signal, cfar_mask, alpha)

#     plt.figure(figsize=(12, 4))
#     plt.plot(np.abs(pure_signal), label="|IQ Signal|", color='blue')
#     plt.plot(threshold_map, label="CFAR Threshold", color='orange', linestyle='--')
#     plt.title(f"Pulse {pulse_num} - CFAR Threshold vs Signal Magnitude")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     detection_map = Spectogram_FunctionsV3.detect_targets_1d(pure_signal, threshold_map)
#     cfar_filtered_signal = pure_signal[detection_map == 1]

#     # Store all relevant info
#     isolated_pulses_data[pulse_num] = {
#         'iq_data': pure_signal,
#         'detection_mask': detection_map,
#         'threshold_map': threshold_map,
#         'cfar_filtered_signal': cfar_filtered_signal
#     }

#     # ---------- Plotting Detection Map ----------
#     plt.figure(figsize=(10, 4))
#     plt.plot(np.abs(pure_signal), label='|IQ Signal|', color='blue')

#     # Scale detection mask to match signal amplitude range
#     scaled_mask = detection_map * np.max(np.abs(pure_signal))
#     plt.plot(scaled_mask, label='CFAR Detection Mask (scaled)', color='green', linestyle='--')

#     plt.title(f"Pulse {pulse_num} - Magnitude with CFAR Detections")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # ---------- Subplot Summary of All Pulses ----------
# if len(isolated_pulses_data) > 0:
#     fig, axes = plt.subplots(len(isolated_pulses_data), 1, figsize=(12, 5 * len(isolated_pulses_data)), sharex=True)
#     if len(isolated_pulses_data) == 1:
#         axes = [axes]

#     for idx, (pulse_num, data_dict) in enumerate(isolated_pulses_data.items()):
#         iq_data = data_dict['iq_data']
#         detection_mask = data_dict['detection_mask']
#         threshold_map = data_dict['threshold_map']

#         ax = axes[idx]
#         ax.plot(np.abs(iq_data), label="|IQ Signal|", color='blue', alpha=0.7)

#         threshold_scaled = threshold_map / np.max(threshold_map) * np.max(np.abs(iq_data))
#         ax.plot(threshold_scaled, label="CFAR Threshold (scaled)", color='orange', linestyle='--', alpha=0.7)

#         detected_indices = np.where(detection_mask == 1)[0]
#         ax.scatter(detected_indices, np.abs(iq_data)[detected_indices],
#                    color='green', marker='o', label='Detections', s=40)

#         ax.set_title(f"Pulse {pulse_num} - CFAR Detection Overview")
#         ax.set_xlabel("Sample Index")
#         ax.set_ylabel("Amplitude")
#         ax.legend()
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()
# else:
#     print("No pulses detected for visualization.")

# # ---------- Optional: Per-Pulse Final Plot ----------
# for pulse_num, data_dict in isolated_pulses_data.items():
#     iq_data = data_dict['iq_data']
#     detection_mask = data_dict['detection_mask']

#     plt.figure(figsize=(10, 4))
#     plt.plot(np.abs(iq_data), label="|IQ Signal|", color='blue', alpha=0.7)

#     detected_indices = np.where(detection_mask == 1)[0]
#     plt.scatter(detected_indices, np.abs(iq_data)[detected_indices],
#                 color='green', label='CFAR Detections', s=30)

#     plt.title(f"Pulse {pulse_num} - Magnitude with CFAR Detections")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# # Print bandwidth results
# print("\nEstimated Bandwidths (MHz) for each pulse:")
# for pulse_num, bw in bandwidth_results.items():
#     if bw is not None:
#         print(f"Pulse {pulse_num}: {bw:.6f} MHz")
#     else:
#         print(f"Pulse {pulse_num}: Bandwidth estimation failed or unavailable.")

