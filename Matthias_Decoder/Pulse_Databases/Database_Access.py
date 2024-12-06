#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import sqlite3
import numpy as np
import polars as pl
import Database_Functions
from sklearn.cluster import DBSCAN
#----------------------------------------------------------------------------------------#
# Path to the SQLite database
db_path = r"/Users/khavishgovind/Documents/Git_Repos/SAR_Radar_RIT/Matthias_Decoder/Pulse_Databases/pulse_characteristics_Mipur.db"

# Connect to the database
conn = sqlite3.connect(db_path)
#----------------------------------------------------------------------------------------#
# Retrieve all pulse characteristics into a Pandas DataFrame
pulse_data_df = pd.read_sql_query(
    """
    SELECT * FROM pulse_data
    """,
    conn
)

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
