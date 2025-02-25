from __future__ import division, print_function, unicode_literals

import numpy as np
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

import matplotlib
matplotlib.use('TkAgg')  # Ensure we're using a GUI backend

import numpy as np
import matplotlib.pyplot as plt



# ------------------- Read the .wav file -------------------
wav_filepath = r"/Users/khavishgovind/Desktop/testData.wav"  # Update this path
fs, radar_data = wavfile.read(wav_filepath)  # Read WAV file

# Convert to float if necessary
if radar_data.dtype != np.float32:
    radar_data = radar_data.astype(np.float32)

# If stereo, take only one channel
if radar_data.ndim > 1:
    radar_data = radar_data[:, 0]  # Taking the first channel

# ------------------- Generate the spectrogram -------------------
fig = plt.figure(figsize=(10, 6), clear=True)
ax = fig.add_subplot(111)
scale = 'dB'

# Generate Spectrogram using Matplotlib's specgram()
Pxx, bb, cc, dd = ax.specgram(
    radar_data, 
    NFFT=2048, 
    Fs=fs , 
    detrend=None, 
    window=np.hanning(2048), 
    scale=scale, 
    noverlap=1024, 
    cmap='Greys'
)

# Add labels and colorbar
plt.colorbar(dd, label='Power (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.show()

print(bb)

# -------------------- Apply Adaptive Threshold -------------------
threshold, Sxx_thresholded = Spectogram_FunctionsV3.adaptive_threshold(Pxx)
print(f"Sxx_thresholded shape: {np.shape(Sxx_thresholded)}")

# -------------------- Apply CFAR Filtering -------------------
vert_guard = 5
vert_avg = 5
hori_guard = 5
hori_avg = 5
alarm_rate = 1e-6

cfar_mask = Spectogram_FunctionsV3.create_2d_mask(vert_guard, vert_avg, hori_guard, hori_avg)
padded_mask = Spectogram_FunctionsV3.create_2d_padded_mask(Pxx, cfar_mask)

#Plot the Padded Mask
plt.figure(figsize=(2, 10))
plt.imshow(padded_mask, interpolation='none', aspect='auto')
plt.title('Vertical CFAR Mask with CUT, Guard Cells, and Averaging Cells')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.colorbar(label='Filter Amplitude')
alpha = Spectogram_FunctionsV3.set_alpha(Spectogram_FunctionsV3.get_total_average_cells(vert_guard, vert_avg, hori_guard, hori_avg), alarm_rate)
thres_map = Spectogram_FunctionsV3.cfar_method(Pxx, padded_mask, alpha)

# Apply CFAR Detection
Sxx_cfar_filtered = Spectogram_FunctionsV3.detect_targets(Pxx, thres_map)

# Plot CFAR Detection Map
plt.figure(figsize=(14, 6))
plt.imshow(Sxx_cfar_filtered, aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label='CFAR Detection')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('CFAR Detection Map')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 5))
# plt.imshow(Sxx_cfar_filtered, interpolation='none', aspect='auto', extent=[cc[0], cc[-1], bb[-1], bb[0]])
# plt.title('Targets')
# plt.xlabel('Time [us]')
# plt.ylabel('Frequency [MHz]')
# plt.colorbar(label='Filter Amplitude')
# plt.tight_layout()

