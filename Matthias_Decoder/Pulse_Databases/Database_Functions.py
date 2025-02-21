#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import polars as pl
import tkinter as tk
from tkinter import ttk
from sklearn.cluster import DBSCAN
#----------------------------------------------------------------------------------------#
# --------------------- Loading Database ---------------------
def load_pulse_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM pulse_data"
    df = pl.read_database(query, conn)
    conn.close()
    return df

def count_rows_in_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pulse_data")
    row_count = cursor.fetchone()[0]
    conn.close()
    return row_count


# --------------------- PDWs Analysis ---------------------
def pdw_analysis(db_path):
    # Connect to the database and load data
    conn = sqlite3.connect(db_path)
    query = "SELECT pulse_number, center_frequency, chirp_rate, pulse_duration FROM pulse_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Perform PDW grouping using pandas
    grouped_df = df.groupby(["center_frequency", "chirp_rate"]).agg(
        pulse_count=("pulse_number", "count"),
        mean_pulse_duration=("pulse_duration", "mean"),
        min_pulse_duration=("pulse_duration", "min"),
        max_pulse_duration=("pulse_duration", "max"),
    ).reset_index()

    pl_grouped_df = pl.from_pandas(grouped_df)

    return pl_grouped_df

# --------------------- Display Database ---------------------
def display_database_in_window(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch data
    cursor.execute("SELECT * FROM pulse_data")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    conn.close()

    # Create window
    root = tk.Tk()
    root.title("Pulse Data Viewer")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    # Scrollbars
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Pack widgets
    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(side="left", fill="both", expand=True)

    # Define column headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Insert data into tree
    for row in rows:
        tree.insert("", "end", values=row)

    root.mainloop()

# --------------------- Plotting Functions ---------------------
# --------------------- I/Q Data Retrieval ---------------------
def retrieve_iq_data_from_db(pulse_number, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT iq_data FROM iq_data WHERE pulse_number = ?"
    cursor.execute(query, (pulse_number,))
    result = cursor.fetchone()
    conn.close()

    if result:
        iq_data = np.frombuffer(result[0], dtype=np.complex64)
        return iq_data
    else:
        print(f"No I/Q data found for pulse number {pulse_number}")
        return None
    
# -------------- Center Frequnecy vs Pulse number ---------------------
def plot_center_frequency_vs_pulse_number(pulse_data):
    plt.figure(figsize=(10, 5))
    plt.plot(pulse_data["pulse_number"], pulse_data["center_frequency"], 'bo-', markersize=5, label="Center Frequency")
    plt.xlabel("Pulse Number")
    plt.ylabel("Center Frequency (Hz)")
    plt.title("Center Frequency vs Pulse Number")
    plt.grid()
    plt.legend()
    plt.show()

# -------------- Bandwidth vs Pulse number ---------------------
def plot_bandwidth_vs_pulse_number(pulse_data):
    plt.figure(figsize=(10, 5))
    plt.plot(pulse_data["pulse_number"], pulse_data["bandwidth"], 'go-', markersize=5, label="Bandwidth")
    plt.xlabel("Pulse Number")
    plt.ylabel("Bandwidth (Hz)")
    plt.title("Bandwidth vs Pulse Number")
    plt.grid()
    plt.legend()
    plt.show()

# -------------- Duration vs Pulse number ---------------------
def plot_duration_vs_pulse_number(pulse_data):
    plt.figure(figsize=(10, 5))
    plt.plot(pulse_data["pulse_number"], pulse_data["pulse_duration"], 'ro-', markersize=5, label="Pulse Duration")
    plt.xlabel("Pulse Number")
    plt.ylabel("Pulse Duration (s)")
    plt.title("Pulse Duration vs Pulse Number")
    plt.grid()
    plt.legend()
    plt.show()

# -------------- Chrip Rate vs Pulse number ---------------------
def plot_chirp_rate_vs_pulse_number(pulse_data):
    plt.figure(figsize=(10, 5))
    plt.plot(pulse_data["pulse_number"], pulse_data["chirp_rate"], 'mo-', markersize=5, label="Chirp Rate")
    plt.xlabel("Pulse Number")
    plt.ylabel("Chirp Rate (Hz/s)")
    plt.title("Chirp Rate vs Pulse Number")
    plt.grid()
    plt.legend()
    plt.show()








                    ### Debugging Functions ###
# --------------------- Plotting I/Q Data  ---------------------
def plot_iq_data(iq_data, pulse_number):
    """
    Plot I/Q data for visualization.
    """
    if iq_data is None:
        print("No I/Q data to plot.")
        return

    plt.figure(figsize=(12, 6))

    # Plot real and imaginary components separately
    plt.subplot(2, 1, 1)
    plt.plot(iq_data.real, 'b', label="In-phase (I)")
    plt.ylabel("Amplitude")
    plt.title(f"I/Q Data for Pulse {pulse_number}")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(iq_data.imag, 'r', label="Quadrature (Q)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()