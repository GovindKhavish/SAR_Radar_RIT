import ee  # Earth Engine API
import numpy as np  # Numerical operations and array manipulation
import matplotlib.pyplot as plt  # Plotting and visualization
from datetime import datetime

# Initialize the Earth Engine module
ee.Initialize(project='ee-govindkhavish-rit')  # Specifying the project linked to my account to access data

# Define a function to expand the area of a polygon by a given factor
def expand_polygon(coords, factor):
    centroid = np.mean(coords, axis=0)
    expanded_coords = [(centroid + (point - centroid) * factor).tolist() for point in coords]
    return expanded_coords

# Define a function to calculate the center of a polygon
def calculate_center(coords):
    centroid = np.mean(coords, axis=0)
    return centroid.tolist()

# Define the coordinates for the Dammam area
dammam_coordinates = [[49.82037320425866, 26.485505016295903],
                      [49.82037320425866, 26.040945171190486],
                      [50.23137546499055, 26.040945171190486],
                      [50.23137546499055, 26.485505016295903],
                      [49.82037320425866, 26.485505016295903]]

# Expand each polygon by a factor of 3
expanded_dammam_coordinates = expand_polygon(np.array(dammam_coordinates), 3)

# Create Earth Engine geometry object for the expanded area
geometry = ee.Geometry.Polygon(expanded_dammam_coordinates)

# Function to generate time series chart
def generate_chart():
    # Generate time series chart for VH polarization over the entire polygon area from the start of the Sentinel-1 mission to the end date
    time_series = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(geometry) \
        .select('VH') \
        .filterDate('2014-04-03', '2021-12-31') \
        .reduce(ee.Reducer.mean()) \
        .reduceRegion(ee.Reducer.mean(), geometry, 500)

    # Get the time series data
    time_series_data = time_series.getInfo()

    # Extract dates and values from the time series data
    dates = []
    values = []
    for key, value in time_series_data.items():
        try:
            # Attempt to convert the key to a timestamp
            timestamp = int(key)
            dates.append(datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d'))
            values.append(value)
        except ValueError:
            # Skip keys that cannot be converted to timestamps
            pass

    # Plot the time series chart
    plt.figure(figsize=(10, 6))
    plt.plot(dates, values, marker='o', linestyle='-')
    plt.title('Radio Frequency Interference Time Series')
    plt.xlabel('Date')
    plt.ylabel('VH Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage: Call generate_chart to generate the time series chart
generate_chart()

