#=========================================================================================
# main_pulse_analysis.py
#Chirp 1 (BW=5.0MHz, FC=-4.0MHz, Duration=10us) injected 123 times
#Chirp 2 (BW=6.0MHz, FC=-2.0MHz, Duration=15us) injected 115 times
#Chirp 3 (BW=7.0MHz, FC=2.0MHz, Duration=20us) injected 110 times
#Chirp 4 (BW=8.0MHz, FC=4.0MHz, Duration=25us) injected 122 times
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import polars as pl
import Database_Functions  # Custom module for database handling and plotting

#----------------------------------------------------------------------------------------#
# Path to the database
db_folder = r"/Users/khavishgovind/Library/CloudStorage/OneDrive-UniversityofCapeTown/Masters/Databases"
#db_folder = r"C:\Users\govin\UCT_OneDrive\OneDrive - University of Cape Town\Masters\Databases"
db_name = "pulse_characteristics_Mipur.db"
# db_name = "pulse_characteristics_Burst_Simulator.db"
db_path = f"{db_folder}/{db_name}"
freq_tolerance = 0.08  # 1% tolerance for center frequency
chirp_tolerance = 0.0147  # 5% tolerance for chirp rate

# Load pulse data
# pulse_data = Database_Functions.load_pulse_data_from_db(db_path)

# ------------------- Tolerance Analysis ---------------------
# results = Database_Functions.analyze_tolerance(db_path)
# print(Database_Functions.print_tolerance_results(results))

# ------------------- Viewing Database ---------------------
# Database_Functions.display_database_in_window(db_path)
# #Database_Functions.display_converted_database_in_window(db_path)

# ------------------- Real Data PDW ---------------------
pdw_results = Database_Functions.pdw_analysis(db_path, freq_tolerance, chirp_tolerance)
#Database_Functions.plot_pdw_bins(pdw_results, tolerance)
Database_Functions.plot_pdw_scatter(pdw_results)
Database_Functions.display_top_bins(pdw_results)
#Database_Functions.plot_top_5_pdw_scatter_with_summary_table(pdw_results,tolerance) 


# ------------------- Plotting ---------------------
# Database_Functions.plot_bandwidth_vs_pulse_number(pulse_data)
# Database_Functions.plot_duration_vs_pulse_number(pulse_data)
# Database_Functions.plot_center_frequency_vs_pulse_number(pulse_data)
# Database_Functions.plot_chirp_rate_vs_pulse_number(pulse_data)


# ------------------- IQ Data Display ---------------------
# pulse_number = 2
# iq_data = Database_Functions.retrieve_iq_data_from_db(pulse_number, db_path)
# Database_Functions.plot_iq_data(iq_data, pulse_number)


