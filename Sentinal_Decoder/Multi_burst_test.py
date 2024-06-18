import sentinel1decoder
import matplotlib.pyplot as plt

# Load the Level 0 file
filepath = "/Users/khavishgovind/Documents/Masters/Data/S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE/"
filename = "s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat"
inputfile = filepath + filename
l0file = sentinel1decoder.Level0File(inputfile)

# Define the bursts you want to visualize
selected_bursts = [4, 6, 8]  # Example: visualize bursts 6, 7, and 8

# Initialize an empty array to accumulate burst data
aggregated_data = None

# Iterate over selected bursts
for selected_burst in selected_bursts:
    # Get the burst data
    burst_data = l0file.get_burst_data(selected_burst)
    
    # Accumulate the burst data
    if aggregated_data is None:
        aggregated_data = burst_data
    else:
        aggregated_data += burst_data

# Plot the aggregated data
plt.figure(figsize=(12, 12))
plt.title("Aggregated Sentinel-1 Raw I/Q Sensor Output")
plt.imshow(abs(aggregated_data), origin='lower')  # Plot the magnitude of the aggregated data
plt.xlabel("Fast Time (down range)")
plt.ylabel("Slow Time (cross range)")
plt.show()