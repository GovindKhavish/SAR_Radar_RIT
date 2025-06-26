#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#----------------------------------------------------------------------------------------#
# -------------------- Adaptive Threshold on Intensity Data -----------------------------#
def adaptive_threshold(array, factor=2):
    mean_value = np.mean(array)
    std_value = np.std(array)
    threshold = mean_value + factor * std_value
    thresholded_array = np.where(array < threshold, 0, array)
        
    return threshold,thresholded_array
#----------------------------------------------------------------------------------------#
#---------------------------- CFAR Function ---------------------------------------------#
def create_2d_mask(vertical_guard_cells, vertical_avg_cells, horizontal_guard_cells, horizontal_avg_cells):
    vertical_size = 2 * (vertical_guard_cells + vertical_avg_cells) + 1
    horizontal_size = 2 * (horizontal_guard_cells + horizontal_avg_cells) + 1
    
    mask = np.zeros((vertical_size, horizontal_size))
    
    center_row = vertical_guard_cells + vertical_avg_cells
    center_col = horizontal_guard_cells + horizontal_avg_cells

    total_avg_cells = (vertical_size*horizontal_size) - ((2*vertical_guard_cells+1)* (horizontal_guard_cells*2 +1))
    
    mask[:center_row - vertical_guard_cells, :] = 1/(total_avg_cells)
    mask[center_row + vertical_guard_cells + 1:, :] = 1/(total_avg_cells)
    mask[:, :center_col - horizontal_guard_cells] = 1/(total_avg_cells)
    mask[:, center_col + horizontal_guard_cells + 1:] = 1/(total_avg_cells)
    
    mask[center_row - vertical_guard_cells:center_row + vertical_guard_cells + 1, 
         center_col - horizontal_guard_cells:center_col + horizontal_guard_cells + 1] = 0
    
    return mask

def get_total_average_cells(vertical_guard_cells, vertical_avg_cells, horizontal_guard_cells, horizontal_avg_cells):
    vertical_size = 2 * (vertical_guard_cells + vertical_avg_cells) + 1
    horizontal_size = 2 * (horizontal_guard_cells + horizontal_avg_cells) + 1
    total_avg_cells = (vertical_size*horizontal_size) - ((2*vertical_guard_cells+1)* (horizontal_guard_cells*2 +1))
    return total_avg_cells

### Padding ###

def create_2d_padded_mask(radar_data, cfar_mask):
    
    radar_rows, radar_cols = radar_data.shape
    mask_rows, mask_cols = cfar_mask.shape

    padded_mask = np.zeros((radar_rows, radar_cols))
    padded_mask[:mask_rows, :mask_cols] = cfar_mask

    return padded_mask

def set_alpha(total_avg_cells,alarm_rate):
    alpha = total_avg_cells*(alarm_rate**(-1/total_avg_cells)-1)
    return alpha

def cfar_method(radar_data, cfar_mask, threshold_multiplier):
    rows, cols = radar_data.shape
    threshold_map = np.zeros_like(radar_data)

    padded_mask = create_2d_padded_mask(radar_data,cfar_mask)

    fft_data = np.fft.fft2(radar_data)
    fft_mask = np.fft.fft2(padded_mask)
    
    fft_threshold = fft_data * fft_mask
    
    threshold_map = np.abs(np.fft.ifft2(np.fft.fftshift(fft_threshold)))
    threshold_map *= threshold_multiplier

    return threshold_map

### Detection ###
def detect_targets(radar_data, threshold_map):
    target_map = np.zeros_like(radar_data)
    len_row, len_col = radar_data.shape 

    for row in range(len_row):
        for col in range(len_col):
            if np.abs(radar_data[row, col]) > threshold_map[row, col]:
                target_map[row, col] = 1
            else:
                target_map[row, col] = 0
    
    return target_map
#-----------------------------------------------------------------------------------------
# --------------------------- CFAR 1D  --------------------------- #
def create_1d_mask(guard_cells, training_cells):
    guard_cells = int(guard_cells)
    training_cells = int(training_cells)
    total_length = 2 * (guard_cells + training_cells) + 1
    mask = np.zeros(total_length)
    
    center = guard_cells + training_cells
    total_training = 2 * training_cells

    mask[:center - guard_cells] = 1 / total_training
    mask[center + guard_cells + 1:] = 1 / total_training

    print(np.sum(mask[mask > 0]) == 1.0)
    
    return mask

def create_1d_padded_mask(signal, cfar_mask):
    signal_len = len(signal)
    mask_len = len(cfar_mask)

    padded_mask = np.zeros(signal_len)
    padded_mask[:mask_len] = cfar_mask

    plt.figure(figsize=(12, 2))
    plt.plot(padded_mask, color='purple', label='Padded CFAR Mask')
    plt.title('Padded 1D CFAR Mask (placed at signal start)')
    plt.xlabel('Sample Index')
    plt.ylabel('Mask Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return padded_mask

def cfar_method_1d(signal, cfar_mask, threshold_multiplier):
    signal_magnitude = np.abs(signal)

    padded_mask = create_1d_padded_mask(signal_magnitude, cfar_mask)
    
    fft_mask = np.fft.fft(np.fft.fftshift(padded_mask))
    fft_signal = np.fft.fft(signal_magnitude)

    fft_product = fft_signal * fft_mask
    threshold_map = np.abs(np.fft.ifft(fft_product))

    threshold_map *= threshold_multiplier

    return threshold_map


def detect_targets_1d(signal, threshold_map):
    signal_magnitude = np.abs(signal)
    return (signal_magnitude > threshold_map).astype(int)

#----------------------------IQ Data Indices ---------------------------------------------#
def spectrogram_to_iq_indices(time_indices, sampling_rate, time_step):
    return (time_indices * time_step * sampling_rate).astype(int)

def spectrogram_time_us_to_iq_index(time_us, fs):
    return (time_us * 1e-6 * fs).astype(int)
#-----------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------#
# === Linear FM Chirp ===
def generate_lfm(fs, bw, fc, duration_us, amplitude=200):
    duration_s = duration_us * 1e-6
    t = np.linspace(0, duration_s, int(fs * duration_s))
    k = bw / duration_s
    phase = 2 * np.pi * (fc * t + 0.5 * k * t**2)
    return amplitude * np.exp(1j * phase)

# === Continuous Wave ===
def generate_cw(fs, fc, duration_us, amplitude=200):
    duration_s = duration_us * 1e-6
    t = np.linspace(0, duration_s, int(fs * duration_s))
    return amplitude * np.exp(1j * 2 * np.pi * fc * t)

# === Barker-coded Pulse ===
def generate_barker(fs, fc, duration_us, code=None, amplitude=200):
    if code is None:
        code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]  # Barker-13
    num_chips = len(code)
    chip_duration_s = (duration_us * 1e-6) / num_chips
    t_chip = np.linspace(0, chip_duration_s, int(fs * chip_duration_s), endpoint=False)
    full_signal = np.concatenate([amplitude * b * np.exp(1j * 2 * np.pi * fc * t_chip) for b in code])
    return full_signal

# === Frank Code ===
def generate_frank(fs, fc, duration_us, n=4, amplitude=200):
    # n: number of rows/cols (Frank matrix size)
    M = n * n
    duration_s = duration_us * 1e-6
    chip_duration = duration_s / M
    t_chip = np.linspace(0, chip_duration, int(fs * chip_duration), endpoint=False)

    full_signal = np.array([], dtype=complex)
    for i in range(n):
        for j in range(n):
            phase = 2 * np.pi * ((i * j) / n)
            chip = amplitude * np.exp(1j * (2 * np.pi * fc * t_chip + phase))
            full_signal = np.concatenate([full_signal, chip])
    return full_signal

# === P4 Polyphase Code ===
def generate_p4(fs, fc, duration_us, n=13, amplitude=200):
    # n: code length
    duration_s = duration_us * 1e-6
    t_chip = np.linspace(0, duration_s / n, int(fs * (duration_s / n)), endpoint=False)
    full_signal = np.array([], dtype=complex)

    for k in range(n):
        phase = np.pi * (k ** 2 / n)
        chip = amplitude * np.exp(1j * (2 * np.pi * fc * t_chip + phase))
        full_signal = np.concatenate([full_signal, chip])
    return full_signal

# === Frequency-Hopping Pulse ===
def generate_frequency_hop(fs, fc_list, duration_us, amplitude=200):
    # fc_list: list of center frequencies for each hop
    num_hops = len(fc_list)
    duration_s = duration_us * 1e-6
    hop_duration = duration_s / num_hops
    t_hop = np.linspace(0, hop_duration, int(fs * hop_duration), endpoint=False)

    signal = np.array([], dtype=complex)
    for fc in fc_list:
        chip = amplitude * np.exp(1j * 2 * np.pi * fc * t_hop)
        signal = np.concatenate([signal, chip])
    return signal

# === Sine-FM (Nonlinear Chirp) ===
def generate_sine_fm(fs, bw, fc, duration_us, mod_freq=5, amplitude=200):
    duration_s = duration_us * 1e-6
    t = np.linspace(0, duration_s, int(fs * duration_s))
    beta = bw / 2
    phase = 2 * np.pi * (fc * t + (beta / mod_freq) * np.cos(2 * np.pi * mod_freq * t))
    return amplitude * np.exp(1j * phase)

# === Multi-Pulse Train ===
def generate_pulse_train(single_pulse_func, num_pulses, pulse_gap_us, **kwargs):
    """
    Repeats a given pulse waveform multiple times with a zero gap in between.
    - single_pulse_func: a function like generate_lfm or generate_cw
    - kwargs: arguments to pass to the waveform function (must include fs, fc, duration_us, etc.)
    """
    pulse = single_pulse_func(**kwargs)
    fs = kwargs['fs']
    gap_samples = int(fs * pulse_gap_us * 1e-6)
    gap = np.zeros(gap_samples, dtype=complex)

    signal = np.array([], dtype=complex)
    for _ in range(num_pulses):
        signal = np.concatenate([signal, pulse, gap])
    return signal


#-----------------------------------------------------------------------------------------#