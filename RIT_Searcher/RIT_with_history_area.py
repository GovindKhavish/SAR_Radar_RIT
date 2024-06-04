# Imports
import ee  # Earth Engine API
import numpy as np  # Numerical operations and array manipulation
import matplotlib.pyplot as plt  # Plotting and visualization
from PIL import Image  # Python Pillow library for images
import urllib.request  # Opening URLs
from datetime import datetime
import pandas as pd  # For handling time series data

# Initialize the Earth Engine module
ee.Initialize(project='ee-govindkhavish-rit')  # Specifying the project linked to my account to access data

# Define a function to expand the area of a polygon by a given factor
def expand_polygon(coords, factor):
    centroid = np.mean(coords, axis=0)
    expanded_coords = [(centroid + (point - centroid) * factor).tolist() for point in coords]
    return expanded_coords

# Define the coordinates for the Dammam area
dammam_coordinates = [[49.82037320425866, 26.485505016295903],
                      [49.82037320425866, 26.040945171190486],
                      [50.23137546499055, 26.040945171190486],
                      [50.23137546499055, 26.485505016295903],
                      [49.82037320425866, 26.485505016295903]]

# Define the coordinates for White Sands Missile Range
white_sands_coordinates = [[-106.507499, 32.497102],
                           [-106.507499, 32.183914],
                           [-106.202217, 32.183914],
                           [-106.202217, 32.497102],
                           [-106.507499, 32.497102]]

# Define the coordinates for Dimona Radar Facility
dimona_coordinates = [[35.027273, 31.049996],
                      [35.027273, 30.932011],
                      [35.158116, 30.932011],
                      [35.158116, 31.049996],
                      [35.027273, 31.049996]]

# Expand each polygon by a factor of 3
expanded_dammam_coordinates = expand_polygon(np.array(dammam_coordinates), 3)
expanded_white_sands_coordinates = expand_polygon(np.array(white_sands_coordinates), 3)
expanded_dimona_coordinates = expand_polygon(np.array(dimona_coordinates), 3)

# Create Earth Engine geometry objects for the expanded areas
geometries = {
    'Dammam': ee.Geometry.Polygon(expanded_dammam_coordinates),
    'White Sands': ee.Geometry.Polygon(expanded_white_sands_coordinates),
    'Dimona': ee.Geometry.Polygon(expanded_dimona_coordinates)
}

# Prompt the user to select a location
location_options = list(geometries.keys())
print("Select a location to view the corresponding images:")
for i, location in enumerate(location_options):
    print(f"{i + 1}. {location}")

location_choice = int(input("Enter the number of your choice: ")) - 1
selected_location = location_options[location_choice]
selected_geometry = geometries[selected_location]

# Prompt the user to enter the start and end dates
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Validate the input dates
try:
    datetime.strptime(start_date, '%Y-%m-%d')
    datetime.strptime(end_date, '%Y-%m-%d')
except ValueError:
    print("Incorrect date format, please enter dates in the format YYYY-MM-DD.")
    exit()

# Function to get image collection for a given geometry and polarization
def get_image_collection(geometry, polarization, orbit):
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))
            .filter(ee.Filter.eq('orbitProperties_pass', orbit))
            .filterDate(start_date, end_date)
            .select([polarization])
            .mean()
            .clip(geometry))

# Function to get image collection for a given geometry and polarization for a time range
def get_image_collection_time_series(geometry, polarization, orbit):
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))
            .filter(ee.Filter.eq('orbitProperties_pass', orbit))
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select([polarization]))

# Function to get the thumbnail URL for an image collection
def get_image_url(image, min_val, max_val, geometry):
    return image.getThumbURL({
        'min': min_val,
        'max': max_val,
        'region': geometry,
        'dimensions': 512
    })

# Get descending and ascending orbit data for VV and VH for the selected location
urls = {}
for orbit in ['DESCENDING', 'ASCENDING']:
    urls[orbit] = {}
    for polarization in ['VV', 'VH']:
        collection = get_image_collection(selected_geometry, polarization, orbit)
        min_val = -25 if polarization == 'VV' else -30
        url = get_image_url(collection, min_val, 0, selected_geometry)
        urls[orbit][polarization] = url

# Open the images from the URLs using Pillow
images = {}
for orbit in urls:
    images[orbit] = {}
    for polarization in urls[orbit]:
        url = urls[orbit][polarization]
        images[orbit][polarization] = Image.open(urllib.request.urlopen(url))

# Convert the images to grayscale and then to NumPy arrays
image_arrays = {}
for orbit in images:
    image_arrays[orbit] = {}
    for polarization in images[orbit]:
        image_gray = images[orbit][polarization].convert('L')
        image_array = np.array(image_gray) / 255.0
        image_arrays[orbit][polarization] = image_array

# Create composite images by averaging the VV and VH images for the selected location
composite_images = {}
for orbit in image_arrays:
    vv_image = image_arrays[orbit]['VV']
    vh_image = image_arrays[orbit]['VH']
    composite_images[orbit] = (vv_image + vh_image) / 2

# Create a final composite image by averaging the descending and ascending composites for the selected location
descending_composite = composite_images['DESCENDING']
ascending_composite = composite_images['ASCENDING']
final_composite_image = (descending_composite + ascending_composite) / 2

# Fetch the data from Earth Engine for VH polarization time series
vh_time_series = get_image_collection_time_series(selected_geometry, 'VH', 'DESCENDING')
vh_data = None
if vh_time_series.size().getInfo() > 0:
    vh_time_series = vh_time_series.reduce(ee.Reducer.mean())
    vh_data = vh_time_series.reduceRegion(ee.Reducer.toList(), selected_geometry, 100).get('VH').getInfo()

# Check if VH data is available
if vh_data:
    # Convert dates to datetime objects
    dates = pd.date_range(start=start_date, end=end_date)

    # Plot the VH polarization intensity over time in a separate window
    plt.figure(figsize=(10, 6))
    plt.plot(dates, vh_data, marker='o', linestyle='-')
    plt.title('VH Polarization Intensity Time Series')
    plt.xlabel('Date')
    plt.ylabel('Intensity')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("VH polarization data not available for the selected location or time range.")

# Plot the images for the selected location in another window
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(selected_location)

axes[0, 0].imshow(image_arrays['DESCENDING']['VV'], cmap='viridis')
axes[0, 0].set_title('Descending Orbit VV')
axes[0, 0].axis('off')

axes[0, 1].imshow(image_arrays['DESCENDING']['VH'], cmap='viridis')
axes[0, 1].set_title('Descending Orbit VH')
axes[0, 1].axis('off')

axes[0, 2].imshow(composite_images['DESCENDING'], cmap='viridis')
axes[0, 2].set_title('Descending Orbit Composite')
axes[0, 2].axis('off')

axes[1, 0].imshow(image_arrays['ASCENDING']['VV'], cmap='viridis')
axes[1, 0].set_title('Ascending Orbit VV')
axes[1, 0].axis('off')

axes[1, 1].imshow(image_arrays['ASCENDING']['VH'], cmap='viridis')
axes[1, 1].set_title('Ascending Orbit VH')
axes[1, 1].axis('off')

axes[1, 2].imshow(composite_images['ASCENDING'], cmap='viridis')
axes[1, 2].set_title('Ascending Orbit Composite')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

