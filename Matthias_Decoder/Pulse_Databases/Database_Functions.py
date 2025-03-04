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
import matplotlib.patches as patches
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

def get_unique_burst_numbers(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT burst_number FROM pulse_data ORDER BY burst_number"
    df = pl.read_database(query, conn)
    conn.close()
    return df["burst_number"].to_list()
# --------------------- PDWs Analysis ---------------------
def bin_values(values, tolerance):
    """Bins values based on a % threshold, checking against all existing bins."""
    values = np.sort(values)
    bins = [values[0]]  # Start with the first value
    
    for val in values[1:]:
        # Check if value fits into any existing bin
        fits_existing_bin = any(abs(val - b) / b <= tolerance for b in bins)
        
        if not fits_existing_bin:  # Create a new bin only if it doesn't fit anywhere
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

def plot_pdw_bins(df, tolerance):
    """Plot rectangular bins for Center Frequency vs. Chirp Rate."""
    
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Define color mapping for unique bins
    unique_bins = df["center_freq_bin"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    color_map = dict(zip(unique_bins, colors))

    for _, row in df.iterrows():
        # Compute bin ranges with 5% tolerance
        freq_min = row["center_freq_bin"] * (1 - tolerance)
        freq_max = row["center_freq_bin"] * (1 + tolerance)
        chirp_min = row["chirp_rate_bin"] * (1 - tolerance)
        chirp_max = row["chirp_rate_bin"] * (1 + tolerance)
        
        # Create a rectangle
        rect = patches.Rectangle(
            (freq_min, chirp_min),  # Bottom-left corner
            freq_max - freq_min,    # Width (frequency range)
            chirp_max - chirp_min,  # Height (chirp rate range)
            linewidth=1.5,
            edgecolor=color_map[row["center_freq_bin"]],
            facecolor="none",
            linestyle="--",
            label=f'Bin {row["center_freq_bin"]}' if row["center_freq_bin"] not in plt.gca().get_legend_handles_labels()[1] else ""
        )
        ax.add_patch(rect)

    # Labels and title
    plt.xlabel("Center Frequency (Hz)")
    plt.ylabel("Chirp Rate (Hz/s)")
    plt.title("Binned Center Frequency vs Chirp Rate (with Tolerance)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Center Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.xlim(df["center_freq_bin"].min() * 0.95, df["center_freq_bin"].max() * 1.05)
    plt.ylim(df["chirp_rate_bin"].min() * 0.95, df["chirp_rate_bin"].max() * 1.05)
    plt.show()



def plot_pdw_scatter(df):
    """Scatter plot of Center Frequency vs. Chirp Rate using the actual values."""

    # Ensure correct column names
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count", "mean_center_frequency", "mean_chirp_rate"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    plt.figure(figsize=(8, 6))

    # Define color mapping for unique frequency bins
    unique_bins = df["center_freq_bin"].unique()
    color_map = plt.cm.get_cmap("tab10", len(unique_bins))  # Using a predefined colormap

    # Scatter plot with actual center frequency and chirp rate
    for i, bin_value in enumerate(unique_bins):
        # Filter data by bin
        bin_data = df[df["center_freq_bin"] == bin_value]
        # Plot data for each bin with a unique color
        plt.scatter(
            bin_data["mean_center_frequency"] / 1e6,  # Convert center frequency to MHz
            bin_data["mean_chirp_rate"]/ 1e12,        # Convert chirp rate to MHz/us
            s=bin_data["pulse_count"] * 5,            # Scale marker size by pulse count
            color=color_map(i),                       # Use the color from colormap
            alpha=0.6
        )

    # Labels and title
    plt.xlabel("Center Frequency (MHz)")
    plt.ylabel("Chirp Rate (MHz/us)")
    plt.title("Center Frequency vs Chirp Rate (Grouped by Tolerance)")
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


def plot_top_5_pdw_scatter_with_summary_table(df, tolerance):
    """Scatter plot of the top 5 groups with the highest pulse counts along with a summary table in the console."""

    print("Columns in DataFrame:", df.columns)  # Debugging step

    # Convert center frequency from Hz to MHz (if it's not already in MHz)
    df["mean_center_frequency"] = df["mean_center_frequency"] / 1e6  # Convert Hz → MHz
    df["mean_chirp_rate"] = df["mean_chirp_rate"] / 1e12 

    # Calculate total pulses for percentage calculation
    total_pulses = df["pulse_count"].sum()

    # Sort by pulse_count in descending order and select the top 5 groups
    top_5_df = df.sort_values(by="pulse_count", ascending=False).head(5)

    # Add percentage column
    top_5_df = top_5_df.assign(percentage=(top_5_df["pulse_count"] / total_pulses) * 100)

    # Ensure correct column names are present in the top 5 DataFrame
    expected_columns = ["center_freq_bin", "chirp_rate_bin", "pulse_count", 
                        "mean_pulse_duration", "min_pulse_duration", "max_pulse_duration", 
                        "mean_center_frequency", "mean_chirp_rate", "percentage"]
    
    for col in expected_columns:
        if col not in top_5_df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # --------- Display Summary Table ---------
    print("\nTop 5 Groups with Most Pulses (Summary Table):")
    print(top_5_df[["mean_center_frequency", "mean_chirp_rate", "pulse_count", "percentage",
                    "mean_pulse_duration", "min_pulse_duration", "max_pulse_duration"]])

    # --------- Plot Scatter ---------
    plt.figure(figsize=(9, 7))

    # Define color mapping for unique frequency bins
    unique_bins = top_5_df["mean_center_frequency"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    color_map = dict(zip(unique_bins, colors))

    # Scatter plot with pulse count as marker size
    for _, row in top_5_df.iterrows():
        center_freq = row["mean_center_frequency"]
        chirp_rate = row["mean_chirp_rate"]
        pulse_count = row["pulse_count"]
        
        # Plot the main scatter point (center point)
        plt.scatter(
            center_freq, chirp_rate,
            s=150, color="black", edgecolors="white", marker="o", zorder=3, label="Center"
        )

        # Draw tolerance box
        rect = plt.Rectangle(
            (center_freq - tolerance, chirp_rate - tolerance),  # Bottom left corner
            2 * tolerance,  # Width
            2 * tolerance,  # Height
            linewidth=1.5, edgecolor=color_map[center_freq], facecolor="none", linestyle="-"
        )
        plt.gca().add_patch(rect)

    # Extend axes limits
    x_min, x_max = top_5_df["mean_center_frequency"].min(), top_5_df["mean_center_frequency"].max()
    y_min, y_max = top_5_df["mean_chirp_rate"].min(), top_5_df["mean_chirp_rate"].max()
    plt.xlim(x_min - 1 * (x_max - x_min), x_max + 1 * (x_max - x_min))  # Extend x-axis by 10%
    plt.ylim(y_min - 1 * (y_max - y_min), y_max + 1 * (y_max - y_min))  # Extend y-axis by 10%

    # Labels and title
    plt.xlabel("Mean Center Frequency (MHz)")  # Updated label
    plt.ylabel("Mean Chirp Rate")
    plt.title("Top 5 Groups with Most Pulses")
    plt.ylabel("Mean Chirp Rate (MHz/µs)")  # Updated label for chirp rate
    plt.title("Top 5 Groups with Most Pulses (MHz Conversion & Highlighted Centers)")

    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    #plt.legend(unique_labels.values(), unique_labels.keys(), title="Top 5 Frequency Groups", fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))

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


def display_converted_database_in_window(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch data
    cursor.execute("SELECT * FROM pulse_data")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    # Convert back to original units
    converted_rows = []
    for row in rows:
        converted_row = list(row)
        converted_row[columns.index("bandwidth")] /= 1e6  # Hz to MHz
        converted_row[columns.index("center_frequency")] /= 1e6  # Hz to MHz
        converted_row[columns.index("chirp_rate")] /= 1e12  # Hz/s to MHz/µs
        converted_row[columns.index("adjusted_start_time")] *= 1e6  # sec to µs
        converted_row[columns.index("adjusted_end_time")] *= 1e6  # sec to µs
        converted_row[columns.index("pulse_duration")] *= 1e6  # sec to µs
        #converted_row[columns.index("toa")] *= 1e6  # sec to µs
        converted_rows.append(tuple(converted_row))

    # Create window
    root = tk.Tk()
    root.title("Converted Pulse Data Viewer")

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

    # Insert converted data into tree
    for row in converted_rows:
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


import sqlite3
import numpy as np

def analyze_tolerance(db_path):
    """
    Analyzes the tolerance of detected radar pulses by comparing them to known injected signals.
    
    Args:
        db_path (str): Path to the SQLite database containing detected pulses.

    Returns:
        dict: A dictionary containing detected counts and percentage errors for each chirp.
    """

    # Known injected signals (ground truth)
    injected_signals = {
        1: {"bandwidth": 5.0, "center_frequency": -4.0, "pulse_duration": 10.0, "count": 116},
        2: {"bandwidth": 6.0, "center_frequency": -2.0, "pulse_duration": 15.0, "count": 121},
        3: {"bandwidth": 7.0, "center_frequency": 2.0, "pulse_duration": 20.0, "count": 108},
        4: {"bandwidth": 8.0, "center_frequency": 4.0, "pulse_duration": 25.0, "count": 125},
    }

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve detected pulses from the database
    cursor.execute("SELECT bandwidth, center_frequency, pulse_duration FROM pulse_data")
    detected_data = cursor.fetchall()
    conn.close()

    # Convert to NumPy array for easier processing
    detected_data = np.array(detected_data)

    # Dictionary to store grouped detected signals
    detected_signals = {key: {"bandwidth": [], "center_frequency": [], "pulse_duration": [], "count": 0} for key in injected_signals}

    # Assign each detected pulse to the closest injected signal using Euclidean distance
    for bw, fc, duration in detected_data:
        closest_key = None
        min_distance = float("inf")

        for key, values in injected_signals.items():
            # Compute Euclidean distance
            distance = np.sqrt(
                (bw - values["bandwidth"])**2 +
                (fc - values["center_frequency"])**2 +
                (duration - values["pulse_duration"])**2
            )

            if distance < min_distance:
                min_distance = distance
                closest_key = key

        # Assign to the closest injected chirp
        if closest_key:
            detected_signals[closest_key]["bandwidth"].append(bw)
            detected_signals[closest_key]["center_frequency"].append(fc)
            detected_signals[closest_key]["pulse_duration"].append(duration)
            detected_signals[closest_key]["count"] += 1

    # Compute percentage deviations
    tolerance_results = {}

    for key, values in detected_signals.items():
        injected = injected_signals[key]

        if values["count"] > 0:
            avg_bw = np.mean(values["bandwidth"])
            avg_fc = np.mean(values["center_frequency"])
            avg_duration = np.mean(values["pulse_duration"])
            detected_count = values["count"]

            tolerance_results[key] = {
                "detected_count": detected_count,
                "bandwidth_error": abs(avg_bw - injected["bandwidth"]) / injected["bandwidth"] * 100,
                "center_frequency_error": abs(avg_fc - injected["center_frequency"]) / abs(injected["center_frequency"]) * 100 if injected["center_frequency"] != 0 else abs(avg_fc) * 100,
                "pulse_duration_error": abs(avg_duration - injected["pulse_duration"]) / injected["pulse_duration"] * 100,
                "count_error": abs(detected_count - injected["count"]) / injected["count"] * 100
            }
        else:
            tolerance_results[key] = {
                "detected_count": 0,
                "bandwidth_error": None,
                "center_frequency_error": None,
                "pulse_duration_error": None,
                "count_error": None
            }

    return tolerance_results

def print_tolerance_results(results):
    print("\n" + "-" * 60)
    print(f"{'Chirp':<6} {'Detected':<10} {'BW Err (%)':<12} {'FC Err (%)':<12} {'Dur Err (%)':<12} {'Count Err (%)':<12}")
    print("-" * 60)

    for key, errors in results.items():
        detected_count = errors['detected_count']

        if errors["bandwidth_error"] is not None:
            bw_error = f"{errors['bandwidth_error']:.2f}"
            fc_error = f"{errors['center_frequency_error']:.2f}"
            dur_error = f"{errors['pulse_duration_error']:.2f}"
            count_error = f"{errors['count_error']:.2f}"
        else:
            bw_error = fc_error = dur_error = count_error = "N/A"

        print(f"{key:<6} {detected_count:<10} {bw_error:<12} {fc_error:<12} {dur_error:<12} {count_error:<12}")

    print("-" * 60 + "\n")


