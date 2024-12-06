#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import sqlite3
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
#----------------------------------------------------------------------------------------#
# -------------------- Fetch I/Q data -----------------------------#
def get_iq_data(pulse_number, conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT iq_data FROM iq_data WHERE pulse_number = ?
        """,
        (pulse_number,)
    )
    result = cursor.fetchone()
    if result:
        iq_data_blob = result[0]
        iq_data = np.frombuffer(iq_data_blob, dtype=complex)
        return iq_data
    else:
        print(f"No I/Q data found for pulse {pulse_number}.")
        return None
#-----------------------------------------------------------------------------------------
