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
def bin_values(values, tolerance):
    """Bins values based on a 5% threshold using numpy."""
    values = np.sort(values)
    bins = [values[0]]  # Start with the first value
    for val in values[1:]:
        if val > bins[-1] * (1 + tolerance):  # If outside the % range, start new bin
            bins.append(val)
    return np.digitize(values, bins, right=True)

def pdw_analysis(db_path, tolerance):
    # Connect to the database and load data
    conn = sqlite3.connect(db_path)
    query = "SELECT pulse_number, center_frequency, chirp_rate, pulse_duration FROM pulse_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Apply binning to group center frequency and chirp rate based on tolerance
    df["center_freq_bin"] = bin_values(df["center_frequency"].values, tolerance)
    df["chirp_rate_bin"] = bin_values(df["chirp_rate"].values, tolerance)

    # Group by these bins and calculate required statistics
    grouped_df = df.groupby(["center_freq_bin", "chirp_rate_bin"]).agg(
        pulse_count=("pulse_number", "count"),
        mean_pulse_duration=("pulse_duration", "mean"),
        min_pulse_duration=("pulse_duration", "min"),
        max_pulse_duration=("pulse_duration", "max"),
        mean_center_frequency=("center_frequency", "mean"),  # Added mean center frequency
        mean_chirp_rate=("chirp_rate", "mean")               # Added mean chirp rate
    ).reset_index()

    # Return the grouped DataFrame
    return grouped_df


def plot_pdw_scatter(df):
    """Scatter plot of Center Frequency vs. Chirp Rate using the actual values."""

    # Ensure correct column names
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    plt.figure(figsize=(8, 6))

    # Define color mapping for unique frequency bins
    unique_bins = df["center_freq_bin"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    color_map = dict(zip(unique_bins, colors))

    # Scatter plot with actual center frequency and chirp rate
    for _, row in df.iterrows():
        plt.scatter(
            row["mean_center_frequency"],  # Use actual center frequency
            row["mean_chirp_rate"],         # Use actual chirp rate
            s=row["pulse_count"] * 5,       # Scale marker size by pulse count
            color=color_map[row["center_freq_bin"]],
            alpha=0.6,
            label=f'Bin {row["center_freq_bin"]}' if row["center_freq_bin"] not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Labels and title
    plt.xlabel("Center Frequency (Hz)")
    plt.ylabel("Chirp Rate (Hz/s)")
    plt.title("Center Frequency vs Chirp Rate (Grouped by 5% Tolerance)")
    plt.legend(title="Center Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


# def plot_top_5_pdw_scatter_with_table(df):
#     """Scatter plot of the top 5 bins with the highest pulse counts along with a summary table in a Tkinter window."""

#     print("Columns in DataFrame:", df.schema)  # Debugging step

#     # Select the top 5 bins with the highest pulse counts
#     top_5_df = df.sort("pulse_count", descending=True).limit(5)

#     # Ensure correct column names
#     expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count", 
#                         "mean_pulse_duration", "min_pulse_duration", "max_pulse_duration"]
#     for col in expected_columns:
#         if col not in top_5_df.columns:
#             raise ValueError(f"Missing expected column: {col}")

#     # --------- Create a Tkinter Window for the Table ---------
#     root = tk.Tk()
#     root.title("Top 5 Bins Summary")

#     frame = tk.Frame(root)
#     frame.pack(fill="both", expand=True)

#     # Treeview widget with scrollbar
#     tree = ttk.Treeview(frame, columns=expected_columns, show="headings")
#     vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
#     hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
#     tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

#     vsb.pack(side="right", fill="y")
#     hsb.pack(side="bottom", fill="x")
#     tree.pack(side="left", fill="both", expand=True)

#     # Define column headings
#     for col in expected_columns:
#         tree.heading(col, text=col)
#         tree.column(col, width=120)

#     # Insert data into treeview
#     for row in top_5_df.iter_rows(named=True):
#         tree.insert("", "end", values=[row[col] for col in expected_columns])

#     # --------- Plot Scatter ---------
#     plt.figure(figsize=(8, 6))

#     # Define color mapping for unique frequency bins
#     unique_bins = top_5_df["center_freq_bin"].unique().to_list()
#     colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
#     color_map = dict(zip(unique_bins, colors))

#     # Scatter plot with pulse count as marker size
#     for row in top_5_df.iter_rows(named=True):
#         plt.scatter(
#             row["center_freq_bin"],
#             row["chirp_rate_bin"],
#             s=row["pulse_count"] * 10,  # Scale marker size by pulse count
#             color=color_map[row["center_freq_bin"]],
#             alpha=0.75,
#             label=f'Bin {row["center_freq_bin"]}'
#         )

#     # Labels and title
#     plt.xlabel("Center Frequency Bin")
#     plt.ylabel("Chirp Rate Bin")
#     plt.title("Top 5 Bins with Most Pulses")
#     plt.legend(title="Top 5 Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.show()

#     root.mainloop()  # Run Tkinter event loop

def plot_top_5_pdw_scatter_with_summary_table(df):
    """Scatter plot of the top 5 bins with the highest pulse counts along with a summary table in the console."""

    print("Columns in DataFrame:", df.columns)  # Debugging step

    # Sort by pulse_count in descending order and select the top 5 groups
    top_5_df = df.sort_values(by="pulse_count", ascending=False).head(5)

    # Ensure correct column names are present in the top 5 DataFrame
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count", 
                        "mean_pulse_duration", "min_pulse_duration", "max_pulse_duration", 
                        "mean_center_frequency", "mean_chirp_rate"]
    
    # Verify that all required columns exist in the DataFrame
    for col in expected_columns:
        if col not in top_5_df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # --------- Display Summary Table ---------
    print("\nTop 5 Bins with Most Pulses (Summary Table):")
    print(top_5_df[["mean_center_frequency", "mean_chirp_rate", "pulse_count", 
                    "mean_pulse_duration", "min_pulse_duration", "max_pulse_duration"]])

    # --------- Plot Scatter ---------
    plt.figure(figsize=(8, 6))

    # Define color mapping for unique frequency bins
    unique_bins = top_5_df["mean_center_frequency"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    color_map = dict(zip(unique_bins, colors))

    # Scatter plot with pulse count as marker size
    for _, row in top_5_df.iterrows():
        plt.scatter(
            row["mean_center_frequency"],  # Plot actual center frequency
            row["mean_chirp_rate"],         # Plot actual chirp rate
            s=row["pulse_count"] * 10,      # Scale marker size by pulse count
            color=color_map[row["mean_center_frequency"]],
            alpha=0.75,
            label=f'Bin {row["mean_center_frequency"]}'
        )

    # Labels and title
    plt.xlabel("Mean Center Frequency")
    plt.ylabel("Mean Chirp Rate")
    plt.title("Top 5 Bins with Most Pulses (with Actual Values)")
    plt.legend(title="Top 5 Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

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