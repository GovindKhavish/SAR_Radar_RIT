#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import os
import sqlite3
import numpy as np
import pandas as pd
import Spectogram_FunctionsV3
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path
#-----------------------------------------------------------------------------------------
# Define the subdirectory path
_simraddir = Path(r'C:\Users\govin\OneDrive\Documents\Git Repositories\Matthias_Decoder\sentinel1decoder (1)\sentinel1decoder')

# Check if the subdirectory exists
if _simraddir.exists():
    sys.path.insert(0, str(_simraddir.resolve()))
    print("Using the right Sentinal Library")
else:
    print(f"Directory {_simraddir} does not exist.")

import sentinel1decoder

# Mipur VH Filepath
filepath = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Mipur_India\S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE"
# filepath = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE/"
filename = '\s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat'

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
burst_array = np.array(echo_bursts['Burst'])
print("Echo Burst Numbers:", burst_array)


for selected_burst in burst_array:
    selection = l0file.get_burst_metadata(selected_burst)

    if selection['Signal Type'].unique()[0] != 0:
        continue 

    print(f"Processing Burst {selected_burst}...")
    radar_data = l0file.get_burst_data(selected_burst)

    #------------------------ Apply CFAR filtering --------------------------------
    start_idx = 0
    end_idx = radar_data.shape[0] - 1 
    fs = 46918402.800000004  

    for idx_n in range(start_idx, end_idx + 1):
        radar_section = radar_data[idx_n, :]
        slow_time_offset = idx_n / fs 

        # ------------------ Spectrogram Data with Local Adaptive Thresholding -------------------
        fig = plt.figure(11, figsize=(6, 6), clear=True)
        ax = fig.add_subplot(111)
        scale = 'dB'
        aa, bb, cc, dd = ax.specgram(radar_data[idx_n, :], NFFT=256, Fs=fs / 1e6, detrend=None, window=np.hanning(256), scale=scale, noverlap=200, cmap='Greys')

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
        # Create a filtered radar data array where values correspond to the non-zero entries of the CFAR mask
        filtered_radar_data = aa * aa_filtered_clean

        # Create a new array to store the filtered spectrogram data (keep values where CFAR mask is non-zero)
        filtered_spectrogram_data = np.zeros_like(aa)  # Initialize with zeros (same shape as aa)
        filtered_spectrogram_data[aa_filtered_clean > 0] = aa[aa_filtered_clean > 0]

        # Label the connected components in the dilated binary mask
        labeled_mask, num_labels = label(aa_filtered_clean, connectivity=2, return_num=True)

        # Define thresholds
        min_angle = 30
        max_angle = 75
        min_diagonal_length = 15
        min_aspect_ratio = 1

        # Create empty mask for valid slashes
        filtered_mask_slashes = np.zeros_like(aa_filtered_clean, dtype=bool)

        # # Get the total number of detected regions
        # num_regions = len(regionprops(labeled_mask))

        # # Stop processing and skip to the next iteration if there are more than 3 regions
        # if num_regions > 3:
        #     continue  # Skip this iteration and move to the next one

        # Main loop to process each region
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
            ransac.fit(x_vals.reshape(-1, 1), y_vals)

            # Get the R² score (how well the line fits)
            r2_score = ransac.score(x_vals.reshape(-1, 1), y_vals)

            # Set a lower R² threshold to allow slight variations
            min_r2_threshold = 0.85

            if r2_score < min_r2_threshold:
                continue  # Skip non-straight shapes

            # If passed all checks, add to final mask
            filtered_mask_slashes[labeled_mask == region.label] = True

        # ---------------------------------------------------------
        time_freq_data = np.column_stack(np.where(filtered_mask_slashes > 0))

        # DBSCAN
        if time_freq_data.shape[0] == 0:
            continue  # Skip if no targets detected
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
                    frequency_indices = bb[cluster_points[:, 0]]
                    time_indices = cc[cluster_points[:, 1]]

                    bandwidth = np.max(frequency_indices) - np.min(frequency_indices)
                    center_frequency = (np.max(frequency_indices) + np.min(frequency_indices)) / 2
                    time_span = np.max(time_indices) - np.min(time_indices)
                    chirp_rate = bandwidth / time_span if time_span != 0 else 0

                    start_time = np.min(time_indices)
                    end_time = np.max(time_indices)

                    adjusted_start_time = start_time + slow_time_offset
                    adjusted_end_time = end_time + slow_time_offset
                    pulse_duration = adjusted_end_time - adjusted_start_time

                    unique_key = (selected_burst, idx_n, cluster_id)  

                    if unique_key not in global_cluster_params:
                        global_cluster_params[unique_key] = []

                    global_cluster_params[unique_key].append({
                        'burst_number': selected_burst,  
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
            # isolated_pulses_data = {}

            # cluster_time_indices = {}
            # for (rangeline_idx, cluster_id), params_list in global_cluster_params.items():
            #     for params in params_list:
            #         start_time_iq = params['start_time_iq']
            #         end_time_iq = params['end_time_iq']
            #         iq_start_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(start_time_iq, sampling_rate, time_step)
            #         iq_end_idx = Spectogram_FunctionsV3.spectrogram_to_iq_indices(end_time_iq,sampling_rate, time_step)
                    
            #         pulse_number = params['pulse_number']
            #         if pulse_number not in cluster_time_indices:
            #             cluster_time_indices[pulse_number] = []
            #         cluster_time_indices[pulse_number].append((rangeline_idx, cluster_id, iq_start_idx, iq_end_idx))

            # for pulse_number in cluster_time_indices:
            #     isolated_pulses_data[pulse_number] = []

            # for idx in range(len(radar_section)):
            #     for pulse_number, clusters in cluster_time_indices.items():
            #         for (rangeline_idx, cluster_id, iq_start_idx, iq_end_idx) in clusters:
            #             if iq_start_idx <= idx <= iq_end_idx:
            #                 if len(isolated_pulses_data[pulse_number]) <= idx:
            #                     isolated_pulses_data[pulse_number].extend([0] * (idx - len(isolated_pulses_data[pulse_number]) + 1))
            #                 isolated_pulses_data[pulse_number][idx] = radar_section[idx]
            #                 break  

            
            # for pulse_number in isolated_pulses_data:
            #     isolated_pulses_data[pulse_number] = np.array(isolated_pulses_data[pulse_number], dtype=complex)

            
            # for pulse_number, data in isolated_pulses_data.items():
            #     if pulse_number not in global_isolated_pulses_data:
            #         global_isolated_pulses_data[pulse_number] = []

            #     global_isolated_pulses_data[pulse_number].append(data) 



# ------------------ Database Storage -------------------
# db_folder = r"/Users/khavishgovind/Documents/Git_Repos/SAR_Radar_RIT/Matthias_Decoder/Pulse_Databases"
db_folder = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Databases"
db_name = "pulse_characteristics_Mipur.db"
db_path = os.path.join(db_folder, db_name)

# Create folder if it doesn't exist
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
    print(f"Folder '{db_folder}' created.")

# Database connection
conn = sqlite3.connect(db_path)

# Prepare the pulse characteristics data for storage
pulse_details = {
    "pulse_number": [],
    "bandwidth": [],
    "center_frequency": [],
    "chirp_rate": [],
    "start_time_index": [],
    "end_time_index": [],
    "adjusted_start_time": [],
    "adjusted_end_time": [],
    "pulse_duration": [],
    "dod": [],           # Set DOD to 0
    "toa": [],           # New parameter: TOA (Adjusted Start Time)
    "aoa": [],           # New parameter: AOA
    "amplitude": [],     # New parameter: Amplitude
    "pos_x": [],         # New parameter: Pos_x
    "pos_y": [],         # New parameter: Pos_y
    "pos_z": [],         # New parameter: Pos_z
    "velo": [],          # New parameter: Velo
    "intra_type": [],    # New parameter: Intra Type
    "sample_rate": [],   # New parameter: Sample Rate
    "intramod": []       # New parameter: Intramod
}

# -------------------- Begin Insertion Loop --------------------

with conn:
    cursor = conn.cursor()
    # Drop the table if it exists, to ensure the schema is correct
    cursor.execute("DROP TABLE IF EXISTS pulse_data")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pulse_data (
        pulse_number INTEGER PRIMARY KEY,
        burst_number INTEGER,
        bandwidth REAL,
        center_frequency REAL,
        chirp_rate REAL,
        start_time_index INTEGER,
        end_time_index INTEGER,
        adjusted_start_time REAL,
        adjusted_end_time REAL,
        pulse_duration REAL,
        dod REAL DEFAULT 0,               
        toa REAL,
        aoa REAL DEFAULT 0,
        amplitude REAL DEFAULT 0,
        pos_x REAL DEFAULT 0,
        pos_y REAL DEFAULT 0,
        pos_z REAL DEFAULT 0,
        velo REAL DEFAULT 0,
        intra_type TEXT DEFAULT 'IW',
        sample_rate REAL,
        intramod REAL DEFAULT 0
    )
    """)
    

    # Create the `iq_data` table for storing I/Q data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS iq_data (
        pulse_number INTEGER,
        iq_data BLOB,
        FOREIGN KEY (pulse_number) REFERENCES pulse_data (pulse_number)
    )
    """)
    # -------------------- Insertion of Pulse Data --------------------
    for unique_key, params_list in global_cluster_params.items():
        for params in params_list:
            # Standard unit conversion
            bandwidth_hz = params["bandwidth"] * 1e6  # MHz to Hz
            center_frequency_hz = params["center_frequency"] * 1e6  # MHz to Hz
            chirp_rate_hz_per_sec = params["chirp_rate"] * 1e6 * 1e6  # MHz/µs to Hz/s
            adjusted_start_time_sec = params["adjusted_start_time"] * 1e-6  # µs to sec
            adjusted_end_time_sec = params["adjusted_end_time"] * 1e-6  # µs to sec
            pulse_duration_sec = params["pulse_duration"] * 1e-6  # µs to sec
            toa_sec = params["adjusted_start_time"] * 1e-6  # µs to sec

            # Insert data into pulse_data table, including burst number
            cursor.execute(
                """INSERT OR REPLACE INTO pulse_data (
                    pulse_number, burst_number, bandwidth, center_frequency, chirp_rate,
                    start_time_index, end_time_index,
                    adjusted_start_time, adjusted_end_time, pulse_duration,
                    dod, toa, aoa, amplitude, pos_x, pos_y, pos_z, velo, intra_type, sample_rate, intramod
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(params["pulse_number"]),  # Global pulse number
                    int(params["burst_number"]),  # Burst number
                    bandwidth_hz, center_frequency_hz, chirp_rate_hz_per_sec,
                    int(params["start_time_index"]), int(params["end_time_index"]),
                    adjusted_start_time_sec, adjusted_end_time_sec, pulse_duration_sec,
                    0,  # Set DOD to 0
                    toa_sec,  # TOA
                    0,  # AOA (default 0)
                    0,  # Amplitude (default 0)
                    0,  # Pos_x (default 0)
                    0,  # Pos_y (default 0)
                    0,  # Pos_z (default 0)
                    0,  # Velo (default 0)
                    "IW",  # Intra Type (default 'IW')
                    fs,  # Sample Rate
                    0  # Intramod (default 0)
                )
            )
    # Commit the changes to the database
    conn.commit()

    # # -------------------- Insertion of I/Q Data --------------------
    # for pulse_number, iq_data_segments in global_isolated_pulses_data.items():
    #     if len(iq_data_segments) == 0:
    #         print(f"Warning: No I/Q data for pulse number {pulse_number}")
    #         continue  # Skip this pulse if no data

    #     for segment in iq_data_segments:
    #         # Serialize the complex I/Q data to binary format (no need for string conversion)
    #         iq_data_blob = segment.tobytes()

    #         # Insert the I/Q data as binary (BLOB) into the database
    #         cursor.execute(
    #             """INSERT OR REPLACE INTO iq_data (pulse_number, iq_data) VALUES (?, ?)""",
    #             (pulse_number, iq_data_blob)
    #         )
            
conn.close()
print(f"Pulse characteristics and I/Q data stored in SQLite3 database at {db_path}.")
