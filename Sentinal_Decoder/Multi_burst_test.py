import sentinel1decoder
import matplotlib.pyplot as plt
import numpy as np

# Load the Level 0 file
filepath = "/Users/khavishgovind/Documents/Masters/Data/S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE/"
filename = "s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat"
inputfile = filepath + filename
l0file = sentinel1decoder.Level0File(inputfile)

# Define the bursts you want to visualize
selected_bursts = [4,7, 6, 8]  # Example: visualize bursts 4, 6, and 8

# Initialize an empty list to store burst data
burst_data_list = []

# Iterate over selected bursts
for selected_burst in selected_bursts:
    # Get the burst data
    burst_data = l0file.get_burst_data(selected_burst)
    
    # Append the burst data to the list
    burst_data_list.append(burst_data)

# Determine the maximum size for padding
max_shape = np.max([data.shape for data in burst_data_list], axis=0)

# Pad each burst data to the maximum shape
padded_burst_data_list = [np.pad(data, ((0, max_shape[0] - data.shape[0]), (0, max_shape[1] - data.shape[1])), mode='constant') for data in burst_data_list]

# Concatenate padded burst data along the slow-time axis (assuming bursts are along slow-time)
combined_data = np.concatenate(padded_burst_data_list, axis=0)
print(combined_data)
print(combined_data.shape)

# Plot the combined data
plt.figure(figsize=(12, 12))
plt.title("Combined Sentinel-1 Raw I/Q Sensor Output")
plt.imshow(abs(combined_data), origin='lower')  # Plot the magnitude of the combined data
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()


