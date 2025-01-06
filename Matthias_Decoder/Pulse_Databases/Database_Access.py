#=========================================================================================
# main_pulse_analysis.py
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import polars as pl
import Database_Functions  # Custom module for database handling and plotting

#----------------------------------------------------------------------------------------#
# Path to the database
db_folder = r"/Users/khavishgovind/Documents/Git_Repos/SAR_Radar_RIT/Matthias_Decoder/Pulse_Databases"
db_name = "pulse_characteristics_Mipur.db"
db_path = f"{db_folder}/{db_name}"

# Load pulse data
pulse_data = Database_Functions.load_pulse_data_from_db(db_path)

# Ensure proper data types
pulse_data["pulse_number"] = pulse_data["pulse_number"].astype(int)

# Optionally convert to a Polars DataFrame
pulse_data_pl = pl.from_pandas(pulse_data_df)

print("All Pulse Characteristics:")
print(pulse_data_pl)

# Fetch specific pulse details
pulse_number = 1428  # Example pulse number
pulse_characteristics = pd.read_sql_query(
    f"""
    SELECT * FROM pulse_data WHERE pulse_number = {pulse_number}
    """,
    conn
)

print(f"Characteristics for Pulse {pulse_number}:")
print(pulse_characteristics)

# Fetch I/Q data for a specific pulse
iq_data = Database_Functions.get_iq_data(pulse_number, conn)
if iq_data is not None:
    print(f"I/Q data for pulse {pulse_number}:")
    print(iq_data)

# Close the connection
conn.close()
