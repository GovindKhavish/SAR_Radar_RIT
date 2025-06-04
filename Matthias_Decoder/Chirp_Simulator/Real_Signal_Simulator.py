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
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

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
filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE/"
filename = 's1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

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
# bb is in MHz and aa is in us as fs/1e6
ax.set_xlabel('Time [us]', fontweight='bold')
ax.set_ylabel('Freq [MHz]', fontweight='bold')
ax.set_title(f'Spectrogram from rangeline {idx_n}', fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------------ Parameters ------------------------
fs = 46918402.8  # Sampling frequency (Hz)
bw = 5e6  # Bandwidth (Hz)
fc = -4e6  # Center frequency offset (Hz)
chirp_duration_us = 20  # Chirp duration in microseconds
chirp_duration_s = chirp_duration_us * 1e-6
row_idx = 560  # Row to inject chirp into

# Time and chirp rate
num_samples = int(chirp_duration_s * fs)
T = num_samples / fs
t = np.linspace(0, T, num_samples)
chirp_rate = bw / chirp_duration_s

# Chirp start and end frequency
start_freq = fc - bw / 2
end_freq = fc + bw / 2

# ---------------- Steep Rise/Fall and Sinusoidal Envelope Mod ----------------
rise_steepness = 10   # Increase this value for sharper rise
fall_steepness = 7    # Keep this the same or adjust separately

rise_len = int(num_samples * 0.05)
flat_len = int(num_samples * 0.8)
fall_len = num_samples - rise_len - flat_len

rise = (np.linspace(0, 1, rise_len)) ** rise_steepness
fall = 1 - (np.linspace(0, 1, fall_len)) ** fall_steepness

flat = np.ones(flat_len)

# Envelope core
envelope = np.concatenate([rise, flat, fall])

# Apply very gentle sinusoidal amplitude modulation (system imperfection)
mod_depth = 0.05  # 3% modulation
mod_freq = 1.5    # cycles across full chirp
modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t / T)
envelope *= modulation

# Normalize
envelope /= np.max(envelope)

# ------------------------ Chirp Signal ------------------------
phase = 2 * np.pi * (start_freq * t + 0.5 * chirp_rate * t**2)
chirp_signal = 50 * envelope * np.exp(1j * phase)

# # Add noise and slight DC bias
# np.random.seed(42)
# chirp_signal += (0.01 * np.random.randn(num_samples) + 1j * 0.01 * np.random.randn(num_samples))

# ------------------------ Inject Into Radar Data ------------------------
num_cols = radar_data.shape[1]
start_idx = num_cols // 2
end_idx = start_idx + num_samples

if end_idx > num_cols:
    raise ValueError("Chirp duration exceeds radar data width.")

radar_data_original = radar_data[row_idx, :].copy()

chirp_signal_padded = np.zeros(num_cols, dtype=complex)
chirp_signal_padded[start_idx:end_idx] = chirp_signal
radar_data[row_idx, :] += chirp_signal_padded

# ------------------------ Overlay Plot ------------------------
plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(np.abs(radar_data[row_idx, :]) + 1e-12), 'r', label='Modified')
plt.plot(10 * np.log10(np.abs(radar_data_original) + 1e-12), 'k', label='Original')
plt.axvline(start_idx, linestyle='--', color='b', label='Chirp Start')
plt.axvline(end_idx, linestyle='--', color='b', label='Chirp End')
plt.xlabel('Fast Time Index')
plt.ylabel('Magnitude (dB)')
plt.title(f'Overlay of Row {row_idx} Before and After Chirp Injection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ Envelope Plot ------------------------
plt.figure(figsize=(8, 4))
plt.plot(t * 1e6, envelope)
plt.title("Realistic Amplitude Envelope (Steep Rise/Fall + Modulation)")
plt.xlabel("Time (µs)")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ IQ Signal Plot ------------------------
plt.figure(figsize=(10, 4))
plt.plot(np.real(chirp_signal), label='I (In-phase)')
plt.plot(np.imag(chirp_signal), label='Q (Quadrature)')
plt.title("Injected Chirp IQ Components (Air Defense Emulation)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
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

aa_filtered_clean = aa_db_filtered
filtered_radar_data = aa * aa_filtered_clean
filtered_spectrogram_data = np.zeros_like(aa)
filtered_spectrogram_data[aa_filtered_clean > 0] = aa[aa_filtered_clean > 0]

dilated_mask = binary_dilation(aa_filtered_clean, footprint=np.ones((1, 1)))
labeled_mask, num_labels = label(dilated_mask, connectivity=2, return_num=True)

min_angle = 10
max_angle = 85
min_diagonal_length = 15
min_aspect_ratio = 1

filtered_mask_slashes = np.zeros_like(dilated_mask, dtype=bool)

plt.figure(figsize=(10, 5))
plt.imshow(dilated_mask, cmap='gray', origin='lower', aspect='auto')
plt.title("Detected Regions and Filtered Slashes")
plt.xlabel("Time (samples)")
plt.ylabel("Frequency (Hz)")

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

    try:
        r2_score = ransac.score(x_vals.reshape(-1, 1), y_vals)
    except ValueError:
        continue  

    min_r2_threshold = 0.85

    if r2_score < min_r2_threshold:
        continue

    filtered_mask_slashes[labeled_mask == region.label] = True

    plt.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], 'r-', linewidth=1)

plt.tight_layout()
plt.show()

# # Display final filtered mask
# plt.figure(figsize=(10, 5))
# plt.imshow(filtered_mask_slashes, cmap='gray', origin='lower', aspect='auto')
# plt.title("Final Filtered Mask")
# plt.xlabel("Time (samples)")
# plt.ylabel("Frequency (Hz)")
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------
print("Starting DBSCAN\n")
# Extract non-zero points as time-frequency data for clustering but are the indices from the spectogram
time_freq_data = np.column_stack(np.where(filtered_mask_slashes > 0))
# DBSCAN Clustering
clusters = DBSCAN(eps=20, min_samples=1).fit_predict(time_freq_data)
frequencies_mhz = bb[time_freq_data[:, 0]]  # frequency in MHz
time_us = cc[time_freq_data[:, 1]]  # Time indices in µs

# plt.figure(figsize=(10, 5))
# plt.scatter(time_us, frequencies_mhz, c=clusters, cmap='viridis', s=5)
# plt.title('DBSCAN Clustering of Chirp Signals')
# plt.xlabel('Time [us]')
# plt.ylabel('Frequency [MHz]')
# plt.colorbar(label='Cluster ID')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.imshow(np.flipud(aa_db_filtered), interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[0], bb[-1]])
# plt.title('Targets')
# plt.xlabel('Time [us]')
# plt.ylabel('Frequency [MHz]')

# for i, cluster in enumerate(np.unique(clusters[clusters != -1])):  # Exclude noise points
#     cluster_points = time_freq_data[clusters == cluster]
#     cluster_time_us = cc[cluster_points[:, 1]]  # Time in us
#     cluster_freq_mhz = bb[cluster_points[:, 0]]  # Frequency in MHz
#     plt.scatter(cluster_time_us, cluster_freq_mhz, c='r', label=f'Cluster {i}', s=5, edgecolors='none', marker='o')

# plt.tight_layout()
# plt.legend()
# plt.show(block=True)

# Number of clusters (excluding noise)
num_clusters = len(np.unique(clusters[clusters != -1]))
print(f"Number of clusters: {num_clusters}")

# ------------------ Start and End Time for Each Cluster -------------------
cluster_time_indices = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Noise

        cluster_points = time_freq_data[clusters == cluster_id]
        sample_indices = [cluster_points[:, 1]]  # Time Samples Indices

        start_time_index = np.min(sample_indices)
        end_time_index = np.max(sample_indices)

        cluster_time_indices[cluster_id] = (start_time_index, end_time_index)

def estimate_bandwidth(iq_data, fs, max_components=10, percentile_thresh=5, coverage=(0.01, 0.99)):

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

# Extract Cluster Parameters
cluster_params = {}

for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # Exclude noise 
        cluster_points = time_freq_data[clusters == cluster_id]
        frequency_indices = bb[cluster_points[:, 0]]
        
        time_indices = [cluster_points[:, 1]]  # Time samples
        time_stamps = cc[cluster_points[:, 1]]  # Time in us
        
        bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
        center_frequency = (np.max(frequency_indices) + np.min(frequency_indices)) / 2
        time_span = np.max(time_stamps) - np.min(time_stamps)  # us
        if time_span != 0:
            chirp_rate = bandwidth / time_span  # MHz per us
        else:
            chirp_rate = 0 
        
        cluster_params[cluster_id] = {
            'bandwidth': bandwidth,
            'center_frequency': center_frequency,
            'chirp_rate': chirp_rate,
            'start_time': np.min(time_stamps),
            'end_time': np.max(time_stamps),
            'start_time_index': np.min(time_indices),
            'end_time_index': np.max(time_indices)
        }

for cluster_id, params in cluster_params.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Bandwidth: {params['bandwidth']} MHz")
    print(f"  Center Frequency: {params['center_frequency']} MHz")
    print(f"  Chirp Rate: {params['chirp_rate']} MHz/us")
    print(f"  Start Time (us): {params['start_time']}")
    print(f"  End Time (us): {params['end_time']}")
    print(f"  Start Time (Index): {params['start_time_index']}")
    print(f"  End Time (Index): {params['end_time_index']}")
    print('---------------------------')

# # Numpy Array conversition
# cluster_params_array = np.array([[cluster_id, params['bandwidth'], params['center_frequency'], params['chirp_rate'], params['start_time_index'], params['end_time_index']]
#                                  for cluster_id, params in cluster_params.items()])

mapped_cluster_indices = {}
for cluster_id, params in cluster_params.items():
    start_time_idx = params['start_time']
    end_time_idx = params['end_time']
    iq_start_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(start_time_idx, fs)
    iq_end_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(end_time_idx, fs)
    mapped_cluster_indices[cluster_id] = (iq_start_idx, iq_end_idx)

isolated_pulses_data = {}
for cluster_id, (iq_start_idx, iq_end_idx) in mapped_cluster_indices.items():
    isolated_pulses_data[cluster_id] = np.zeros_like(radar_section, dtype=complex)
    for idx in range(len(radar_section)):
        if iq_start_idx <= idx <= iq_end_idx:
            isolated_pulses_data[cluster_id][idx] = radar_section[idx]

# ------------------ GMM Bandwidth Estimation ------------------
gmm_bandwidths = {}
for cluster_id, iq_data in isolated_pulses_data.items():
    if np.any(iq_data):
        print(f"\nEstimating bandwidth for Cluster {cluster_id} using GMM:")
        gmm_bw = estimate_bandwidth(iq_data, fs)
        gmm_bandwidths[cluster_id] = gmm_bw
    else:
        print(f"\nCluster {cluster_id} has no valid I/Q data. Skipping GMM bandwidth estimation.")

print("\n=== Summary of GMM-Based Bandwidths ===")
for cluster_id in gmm_bandwidths:
    print(f"Cluster {cluster_id} - GMM Estimated Bandwidth: {gmm_bandwidths[cluster_id]:.3f} MHz")

# ------------------ Visualization ------------------
if len(isolated_pulses_data) > 0:
    fig, axes = plt.subplots(len(isolated_pulses_data), 1, figsize=(10, 6), sharex=True, sharey=True)
    if len(isolated_pulses_data) == 1:
        axes = [axes]

    for idx, (cluster_id, iq_data) in enumerate(isolated_pulses_data.items()):
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

plt.figure(figsize=(12, 4))
plt.plot(np.abs(radar_section), label='IQ Magnitude')
plt.axvline(iq_start_idx, color='r', linestyle='--', label='Mapped Start')
plt.axvline(iq_end_idx, color='g', linestyle='--', label='Mapped End')
plt.legend()
plt.title('IQ Data with Mapped Spectrogram Time Indices')
plt.show()



# NFFT = 256
# noverlap = 200
# sampling_rate = fs
# time_step = (NFFT - noverlap) / sampling_rate  # seconds

# mapped_pulse_indices = {}

# for global_pulse_number, param_list in global_cluster_params.items():
#     params = param_list[0]
#     start_time_idx = params['start_time']
#     end_time_idx = params['end_time']
    
#     iq_start_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(start_time_idx, sampling_rate)
#     iq_end_idx = Spectogram_FunctionsV3.spectrogram_time_us_to_iq_index(end_time_idx, sampling_rate)

#     mapped_pulse_indices[global_pulse_number] = (iq_start_idx, iq_end_idx)

# isolated_pulses_data = {}
# bandwidth_results = {}

# for pulse_num, (iq_start_idx, iq_end_idx) in mapped_pulse_indices.items():
#     # Isolate only the pure pulse I/Q data segment
#     extension = 1000  # samples to extend on both sides

#     start_idx = max(0, iq_start_idx - extension)
#     end_idx = min(len(radar_section), iq_end_idx + extension + 1)

#     pure_signal = radar_section[start_idx:end_idx]

#     #pure_signal = radar_section[iq_start_idx:iq_end_idx+1]

#     # Store for plotting if needed
#     isolated_pulses_data[pulse_num] = pure_signal
    
#     # Estimate bandwidth on pure signal only (no zero-padding)
#     bw = estimate_bandwidth(pure_signal, sampling_rate)
#     bandwidth_results[pulse_num] = bw


# # Your existing plot code for real and imaginary parts
# if len(isolated_pulses_data) > 0:
#     fig, axes = plt.subplots(len(isolated_pulses_data), 1, figsize=(10, 6), sharex=True, sharey=True)
#     if len(isolated_pulses_data) == 1:
#         axes = [axes]

#     for idx, (pulse_num, iq_data) in enumerate(isolated_pulses_data.items()):
#         axes[idx].plot(np.real(iq_data), label=f"Pulse {pulse_num} - Real", color='blue')
#         axes[idx].plot(np.imag(iq_data), label=f"Pulse {pulse_num} - Imag", color='red')
#         axes[idx].set_title(f"Pulse {pulse_num} - Isolated I/Q Data")
#         axes[idx].set_xlabel("Sample Index")
#         axes[idx].set_ylabel("Amplitude")
#         axes[idx].legend()

#     plt.tight_layout()
#     plt.show()

#     # New figure for absolute values of I/Q data
#     fig_abs, axes_abs = plt.subplots(len(isolated_pulses_data), 1, figsize=(10, 6), sharex=True)
#     if len(isolated_pulses_data) == 1:
#         axes_abs = [axes_abs]

#     for idx, (pulse_num, iq_data) in enumerate(isolated_pulses_data.items()):
#         axes_abs[idx].plot(np.abs(iq_data), label=f"Pulse {pulse_num} - |I/Q|", color='green')
#         axes_abs[idx].set_title(f"Pulse {pulse_num} - Magnitude of I/Q Data")
#         axes_abs[idx].set_xlabel("Sample Index")
#         axes_abs[idx].set_ylabel("Magnitude")
#         axes_abs[idx].legend()

#     plt.tight_layout()
#     plt.show()

# else:
#     print("No pulses detected for visualization.")

# # Print bandwidth results summary
# print("\nEstimated Bandwidths (MHz) for each pulse:")
# for pulse_num, bw in bandwidth_results.items():
#     if bw is not None:
#         print(f"Pulse {pulse_num}: {bw:.6f} MHz")
#     else:
#         print(f"Pulse {pulse_num}: Bandwidth estimation failed or unavailable.")
