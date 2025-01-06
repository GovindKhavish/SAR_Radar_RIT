#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import polars as pl
import tkinter as tk
from tkinter import ttk
from sklearn.cluster import DBSCAN
#----------------------------------------------------------------------------------------#
# --------------------- I/Q Data Retrieval ---------------------
def get_iq_data(db_path, pulse_number):
    conn = sqlite3.connect(db_path)
    query = f"SELECT iq_data FROM iq_data WHERE pulse_number = {pulse_number}"
    cursor = conn.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    if result:
        iq_data = np.frombuffer(result[0], dtype=np.complex64)  # Assuming complex64 format
        return iq_data
    return None

# --------------------- Database Loading ---------------------
def load_pulse_data_from_db(db_path):
    """
    Loads pulse data from the SQLite database and returns a Pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT pulse_number, bandwidth, center_frequency, pulse_duration FROM pulse_data"
    pulse_data_df = pd.read_sql(query, conn)
    conn.close()
    return pulse_data_df


def display_database_in_window(db_path):
    # Load database using SQLite3 and Pandas
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM pulse_data"
    pulse_data_df = pd.read_sql(query, conn)
    conn.close()

    # Create the main tkinter window
    root = tk.Tk()
    root.title("Database Viewer")

    # Add a Treeview widget to display the data
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(frame, columns=list(pulse_data_df.columns), show='headings')

    # Define column headings
    for col in pulse_data_df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER, width=100)

    # Insert data into the Treeview widget
    for index, row in pulse_data_df.iterrows():
        tree.insert("", tk.END, values=list(row))

    tree.pack(fill=tk.BOTH, expand=True)

    # Add a vertical scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Function to display selected row details
    def show_selected_row_details():
        selected_item = tree.focus()  # Get selected item ID
        if not selected_item:
            return

        values = tree.item(selected_item, 'values')  # Get the row data

        # Create a new window to display details
        details_window = tk.Toplevel(root)
        details_window.title("Row Details")

        # Display row details
        for i, col in enumerate(pulse_data_df.columns):
            label = ttk.Label(details_window, text=f"{col}: {values[i]}")
            label.pack(anchor=tk.W, padx=10, pady=5)

    # Add a button to show row details
    button = ttk.Button(root, text="Show Selected Row Details", command=show_selected_row_details)
    button.pack(pady=10)

    # Run the tkinter event loop
    root.mainloop()

# --------------------- Plotting Functions ---------------------
def plot_bandwidth_vs_pulse_number(dataframe, title="Bandwidth vs. Pulse Number"):
    """
    Plots Bandwidth vs Pulse Number.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe["pulse_number"], dataframe["bandwidth"], marker='o', linestyle='-', color='b')
    plt.xlabel("Pulse Number")
    plt.ylabel("Bandwidth (Hz)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_duration_vs_pulse_number(dataframe, title="Pulse Duration vs. Pulse Number"):
    """
    Plots Pulse Duration vs Pulse Number.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe["pulse_number"], dataframe["pulse_duration"], marker='o', linestyle='-', color='g')
    plt.xlabel("Pulse Number")
    plt.ylabel("Pulse Duration (s)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_center_frequency_vs_pulse_number(dataframe, title="Center Frequency vs. Pulse Number"):
    """
    Plots Center Frequency vs Pulse Number.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe["pulse_number"], dataframe["center_frequency"], marker='o', linestyle='-', color='r')
    plt.xlabel("Pulse Number")
    plt.ylabel("Center Frequency (Hz)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def retrieve_iq_data_from_db(pulse_number, db_path):
    """
    Retrieves I/Q data for a specific pulse from the SQLite database.
    
    Args:
        pulse_number (int): The pulse number for which to retrieve the I/Q data.
        db_path (str): Path to the SQLite database.
        
    Returns:
        np.ndarray: The deserialized I/Q data as a numpy array.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Retrieve the I/Q data as a BLOB from the database
    cursor.execute("SELECT iq_data FROM iq_data WHERE pulse_number=?", (pulse_number,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result is None:
        print(f"No I/Q data found for pulse {pulse_number}.")
        return None
    
    # Deserialize the I/Q data from the BLOB (binary data)
    iq_data_blob = result[0]
    iq_data = np.frombuffer(iq_data_blob, dtype=complex)
    
    return iq_data


def plot_iq_data(iq_data, pulse_number):
    """
    Plots the I/Q data for a specific pulse.
    
    Args:
        iq_data (np.ndarray): The deserialized I/Q data for the pulse (complex array).
        pulse_number (int): The pulse number for the plot title.
    """
    # Ensure there is I/Q data
    if iq_data is None:
        print(f"No I/Q data found for pulse {pulse_number}.")
        return
    
    # Time axis (based on the number of samples)
    time = np.arange(len(iq_data))
    
    # Plotting the I/Q data (Real and Imaginary parts)
    plt.figure(figsize=(12, 6))
    plt.plot(time, np.real(iq_data), label=f"Pulse {pulse_number} - Real (I)", color='blue')
    plt.plot(time, np.imag(iq_data), label=f"Pulse {pulse_number} - Imaginary (Q)", color='red')

    # Title and labels
    plt.title(f"Pulse {pulse_number} - I/Q Data")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()




