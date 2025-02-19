#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
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

#import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Specify the path to your .mat file
file_path = '/Users/khavishgovind/Documents/Git_Repos/SAR_Radar_RIT/Matthias_Decoder/Matlab_Simulator/matlab_spectrogram.mat'  # Change this to the actual path

# Load the MATLAB spectrogram data from the .mat file
mat_data = scipy.io.loadmat(file_path)

# Extract the spectrogram data (in dB) from the file
S_dB_matlab = mat_data['S_dB']
T_matlab = mat_data['T']
F_centered_matlab = mat_data['F_centered']

# In your Python script, calculate the spectrogram using matplotlib (same parameters)
idx_n = 1200  # Row index for Python
radar_section = radar_data[idx_n, :]
fs = 46918402.8  # Sampling frequency in Hz

# Create a spectrogram using matplotlib
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the spectrogram with matplotlib parameters
aa, bb, cc, dd = ax.specgram(radar_section, NFFT=256, Fs=fs / 1e6, window=np.hanning(256), noverlap=200, scale='dB', cmap='Greys')

# Get the spectrogram data (dB scale)
S_dB_python = 10 * np.log10(np.abs(cc) ** 2)  # This should be 2D

# Ensure S_dB_python is 2D (time, frequency)
print(f"S_dB_python shape: {S_dB_python.shape}")  # Should be (time, frequency)

# Convert frequency axis to MHz (same as MATLAB)
F_python = bb  # Frequency in Hz
F_python_MHz = F_python / 1e6

# Create a symmetric frequency axis (both positive and negative frequencies)
F_python_centered = np.concatenate([-np.flipud(F_python[1:]), F_python])

# Now compare the spectrogram matrices
# First, compare the time axis: Ensure T and T_matlab match
print(f"Max time difference: {np.max(np.abs(aa - T_matlab))}")

# Now, compare the frequency axis: Ensure F and F_centered match
print(f"Max frequency difference: {np.max(np.abs(F_python_centered - F_centered_matlab))}")

# Normalize both spectrogram matrices for a fair comparison
S_dB_python_normalized = (S_dB_python - np.min(S_dB_python)) / (np.max(S_dB_python) - np.min(S_dB_python))
S_dB_matlab_normalized = (S_dB_matlab - np.min(S_dB_matlab)) / (np.max(S_dB_matlab) - np.min(S_dB_matlab))

# Compute the maximum difference between the spectrogram matrices
difference = np.max(np.abs(S_dB_python_normalized - S_dB_matlab_normalized))
print(f"Maximum difference between spectrograms: {difference}")

# Plot Python spectrogram
plt.figure(figsize=(6, 6))
plt.imshow(S_dB_python, aspect='auto', origin='lower', extent=[aa.min(), aa.max(), F_python_centered.min(), F_python_centered.max()], cmap='Greys')
plt.title('Python Spectrogram')
plt.xlabel('Time [us]')
plt.ylabel('Freq [MHz]')
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot MATLAB spectrogram
plt.figure(figsize=(6, 6))
plt.imshow(S_dB_matlab, aspect='auto', origin='lower', extent=[T_matlab.min(), T_matlab.max(), F_centered_matlab.min(), F_centered_matlab.max()], cmap='Greys')
plt.title('MATLAB Spectrogram')
plt.xlabel('Time [us]')
plt.ylabel('Freq [MHz]')
plt.colorbar()
plt.tight_layout()