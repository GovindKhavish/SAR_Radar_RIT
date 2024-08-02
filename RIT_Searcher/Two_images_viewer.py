# Basic test program to view two GRD images from Sentinel-1 (ascending and descending) in VSCode using Python and Google Earth Engine
# Version 0.2
# Modified: 2024/05/25

# Imports
import ee  # Earth Engine API
import numpy as np  # Numerical operations and array manipulation
import matplotlib.pyplot as plt  # Plotting and visualization
from PIL import Image  # Python Pillow library for images
import urllib.request  # Opening URLs

# Initialize the Earth Engine module
ee.Initialize(project='ee-govindkhavish-rit')  # Specifying the project linked to my account to access data

# Define the coordinates for the Dammam area
dammam_coordinates = [[49.82037320425866, 26.485505016295903],
                      [49.82037320425866, 26.040945171190486],
                      [50.23137546499055, 26.040945171190486],
                      [50.23137546499055, 26.485505016295903],
                      [49.82037320425866, 26.485505016295903]]

# Create an Earth Engine geometry object for the Dammam area
dammam_geometry = ee.Geometry.Polygon(dammam_coordinates)

# Define the date range
start_date = '2021-12-01'
end_date = '2021-12-31'

# Collect descending orbit data
collection_descending = (ee.ImageCollection('COPERNICUS/S1_GRD')
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                         .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                         .filterDate(start_date, end_date)
                         .select(['VV'])
                         .mean())

# Collect ascending orbit data
collection_ascending = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                        .filterDate(start_date, end_date)
                        .select(['VV'])
                        .mean())

# Get the image URL for the descending orbit
url_descending = collection_descending.clip(dammam_geometry).getThumbURL({
    'min': -25,
    'max': 0,
    'region': dammam_geometry,
    'dimensions': 512
})

# Get the image URL for the ascending orbit
url_ascending = collection_ascending.clip(dammam_geometry).getThumbURL({
    'min': -25,
    'max': 0,
    'region': dammam_geometry,
    'dimensions': 512
})

# Open the images from the URLs using Pillow
image_descending = Image.open(urllib.request.urlopen(url_descending))
image_ascending = Image.open(urllib.request.urlopen(url_ascending))

# Convert the images to grayscale
image_descending_gray = image_descending.convert('L')
image_ascending_gray = image_ascending.convert('L')

# Convert the grayscale images to NumPy arrays
image_descending_array = np.array(image_descending_gray)
image_ascending_array = np.array(image_ascending_gray)

# Normalize the images to the range [0, 1]
image_descending_normalized = image_descending_array / 255.0
image_ascending_normalized = image_ascending_array / 255.0

# Create a composite image by averaging the two normalized images
composite_image = (image_descending_normalized + image_ascending_normalized) / 2

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the descending orbit image
axes[0].imshow(image_descending_normalized, cmap='viridis')
axes[0].set_title('Descending Orbit')
axes[0].axis('off')

# Display the ascending orbit image
axes[1].imshow(image_ascending_normalized, cmap='viridis')
axes[1].set_title('Ascending Orbit')
axes[1].axis('off')

# Display the composite image
axes[2].imshow(composite_image, cmap='viridis')
axes[2].set_title('Composite Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()
