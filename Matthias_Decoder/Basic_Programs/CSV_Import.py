import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = '/Users/khavishgovind/Documents/Git_Repos/SAR_Radar_RIT/Matthias_Decoder/Basic_Programs/Csv_Files/Mipur_radar_data.csv'

# Read the CSV file
df = pd.read_csv(file_path, header=None)

# Convert all entries to complex numbers
df_complex = df.map(lambda x: complex(x))

# Convert complex numbers to their magnitude (absolute value)
df_mag = np.abs(df_complex.values)

# Plot the magnitude of the radar data
plt.figure(figsize=(14, 6))
plt.imshow(10 * np.log10(df_mag), aspect='auto', interpolation='none', origin='lower')
#plt.imshow(df_mag, aspect='auto', interpolation='none', origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Fast Time')
plt.ylabel('Slow Time')
plt.title('Radar Data Magnitude from CSV')
plt.show()
