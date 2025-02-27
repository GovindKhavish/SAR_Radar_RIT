#=========================================================================================
# _common_imports_v3_py.py ]
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------
import sys
from pathlib import Path
#-----------------------------------------------------------------------------------------

import sentinel1decoder


data = {
    "Mipur": {
        "VH": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Mipur_India/S1A_IW_RAW__0SDV_20220115T130440_20220115T130513_041472_04EE76_AB32.SAFE",
            "filename": '/s1a-iw-raw-s-vh-20220115t130440-20220115t130513-041472-04ee76.dat',
        }
    },
    "Damascus": {
        "VH": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Lebanon/S1A_IW_RAW__0SDV_20190219T033515_20190219T033547_025993_02E57A_C90C.SAFE",
            "filename": '/s1a-iw-raw-s-vh-20190219t033515-20190219t033547-025993-02e57a.dat',
        },
        "VV": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Damascus_Lebanon/S1A_IW_RAW__0SDV_20190219T033515_20190219t033547_025993_02E57A_C90C.SAFE",
            "filename": '/s1a-iw-raw-s-vv-20190219t033515-20190219t033547-025993-02e57a.dat',
        },
    },
    "Rostov": {
        "VH": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Rostov_Russia/S1A_IW_RAW__0SDV_20210722T151210_20210722T151242_038892_0496D3_23F3.SAFE",
            "filename": '/s1a-iw-raw-s-vh-20210722t151210-20210722t151242-038892-0496d3.dat',
        },
        "VV": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Rostov_Russia/S1A_IW_RAW__0SDV_20210722T151210_20210722T151242_038892_0496D3_23F3.SAFE",
            "filename": '/s1a-iw-raw-s-vv-20210722t151210-20210722t151242-038892-0496d3.dat',
        },
    },
    "Guam": {
        "VH": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Guam_USA/2021_09_06/S1B_IW_RAW__0SDV_20210906T200756_20210906T200822_028582_036935_3C18.SAFE",
            "filename": '/s1b-iw-raw-s-vh-20210906t200756-20210906t200822-028582-036935.dat',
        },
        "VV": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Guam_USA/2021_09_06/S1B_IW_RAW__0SDV_20210906T200756_20210906T200822_028582_036935_3C18.SAFE",
            "filename": '/s1b-iw-raw-s-vv-20210906t200756-20210906t200822-028582-036935.dat',
        },
    },
    "Nazareth": {
        "VH": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Nazareth_Isreal/S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE",
            "filename": '/s1a-iw-raw-s-vh-20190224t034343-20190224t034416-026066-02e816.dat',
        },
        "VV": {
            "filepath": r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Data/Nazareth_Isreal/S1A_IW_RAW__0SDV_20190224T034343_20190224T034416_026066_02E816_A557.SAFE",
            "filename": '/s1a-iw-raw-s-vv-20190224t034343-20190224t034416-026066-02e816.dat',
        },
    },
    "Beruit": {
        "VH": {
            "filepath": r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Beirut_Lebanon\S1A_IW_RAW__0SDV_20241006T154123_20241006T154155_055984_06D89C_05A2.SAFE",
            "filename": '\s1a-iw-raw-s-vh-20241006t154123-20241006t154155-055984-06d89c.dat',
        },
        "VV": {
            "filepath": r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Beirut_Lebanon\S1A_IW_RAW__0SDV_20241006T154123_20241006T154155_055984_06D89C_05A2.SAFE",
            "filename": '\s1a-iw-raw-s-vv-20241006t154123-20241006t154155-055984-06d89c.dat',
        },
    },
    "Augsberg": {
        "VH": {
            "filepath": r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Augsburg_Germany\S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE",
            "filename": '\s1a-iw-raw-s-vh-20190219t033540-20190219t033612-025993-02e57a.dat',
        },
        "VV": {
            "filepath": r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Data\Augsburg_Germany\S1A_IW_RAW__0SDV_20190219T033540_20190219T033612_025993_02E57A_771F.SAFE",
            "filename": '\s1a-iw-raw-s-vv-20190219t033540-20190219t033612-025993-02e57a.dat',
        },
    },
}

print("Select a location: Mipur, Damascus, Rostov, Guam, Nazareth, Beruit,Augsberg")
location = input("Enter location: ").strip()
print("Select polarization: VH or VV")
polarization = input("Enter polarization: ").strip().upper()

try:
    selected = data[location][polarization]
    filepath = selected['filepath']
    filename = selected['filename']
except KeyError:
    print("Invalid location or polarization. Exiting...")
    sys.exit(1)

inputfile = filepath + filename

l0file = sentinel1decoder.Level0File(inputfile)

# # Identify valid bursts with "Signal Type == 0"
# echo_bursts = l0file.burst_info[l0file.burst_info['Signal Type'] == 0]
# valid_burst_numbers = echo_bursts['Burst'].tolist()

total_bursts = len(l0file.burst_info) 
valid_burst_numbers = []

for burst_number in range(1, total_bursts + 1):
    metadata = l0file.get_burst_metadata(burst_number)
    if metadata['Signal Type'].unique()[0] == 0:
        valid_burst_numbers.append(burst_number)

print(f"Valid burst numbers: {valid_burst_numbers}")


for burst_num in valid_burst_numbers:
    radar_data = l0file.get_burst_data(burst_num)

    plt.figure(figsize=(14, 6))
    plt.imshow(10 * np.log10(abs(radar_data[:, :])), aspect='auto', interpolation='none', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Fast Time')
    plt.ylabel('Slow Time')
    plt.title(f'Burst {burst_num} Data')
    plt.show()
    plt.clf() 
    plt.close()