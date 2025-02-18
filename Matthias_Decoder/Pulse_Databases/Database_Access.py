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
db_folder = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Databases"
# db_folder = r"C:\Users\govin\OneDrive\Documents\Databases"
db_name = "pulse_characteristics_WhiteSands.db"
db_path = f"{db_folder}/{db_name}"

# Load pulse data
pulse_data = Database_Functions.load_pulse_data_from_db(db_path)

# View the database in a window and allow row selection
Database_Functions.display_database_in_window(db_path)

# Plot the pulse characteristics
rows = Database_Functions.count_rows_in_database(db_path)
print(rows)
Database_Functions.plot_bandwidth_vs_pulse_number(pulse_data)
Database_Functions.plot_duration_vs_pulse_number(pulse_data)
Database_Functions.plot_center_frequency_vs_pulse_number(pulse_data)
Database_Functions.plot_chirp_rate_vs_pulse_number(pulse_data)

# # If I/Q data visualization or specific pulse retrieval is needed
# pulse_number = 2
# iq_data = Database_Functions.retrieve_iq_data_from_db(pulse_number, db_path)
# Database_Functions.plot_iq_data(iq_data, pulse_number)


