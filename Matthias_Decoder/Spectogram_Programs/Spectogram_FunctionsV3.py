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
#-----------------------------------------------------------------------------------------