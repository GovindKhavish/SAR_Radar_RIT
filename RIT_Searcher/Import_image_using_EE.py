import ee
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Initialize the Earth Engine module
ee.Initialize(project='ee-govindkhavish-rit')

# Define the date
date = "2021-12-13"

# Define the coordinates for the Dammam area
dammam_coordinates = [[49.82037320425866, 26.485505016295903],
                      [49.82037320425866, 26.040945171190486],
                      [50.23137546499055, 26.040945171190486],
                      [50.23137546499055, 26.485505016295903],
                      [49.82037320425866, 26.485505016295903]]

# Load the Sentinel-1 GRD data for ascending passes
ascending_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filterBounds(ee.Geometry.Polygon(dammam_coordinates))
                        .filterDate(date)
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

# Load the Sentinel-1 GRD data for descending passes
descending_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                         .filterBounds(ee.Geometry.Polygon(dammam_coordinates))
                         .filterDate(date)
                         .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

# Select the polarization bands and convert to decibels for ascending passes
ascending_vv = ascending_collection.select('VV').map(lambda img: img.select('VV').log10().multiply(10))
ascending_vh = ascending_collection.select('VH').map(lambda img: img.select('VH').log10().multiply(10))

# Select the polarization bands and convert to decibels for descending passes
descending_vv = descending_collection.select('VV').map(lambda img: img.select('VV').log10().multiply(10))
descending_vh = descending_collection.select('VH').map(lambda img: img.select('VH').log10().multiply(10))

# Merge the ascending VV and VH collections
ascending_image = ascending_vv.mean().addBands(ascending_vh.mean())

# Merge the descending VV and VH collections
descending_image = descending_vv.mean().addBands(descending_vh.mean())

# Get thumbnail URLs
ascending_thumb_url = ascending_image.getThumbURL({'min': -25, 'max': 5})
descending_thumb_url = descending_image.getThumbURL({'min': -25, 'max': 5})

# Read images from URLs
response_ascending = requests.get(ascending_thumb_url)
ascending_img = Image.open(BytesIO(response_ascending.content))

response_descending = requests.get(descending_thumb_url)
descending_img = Image.open(BytesIO(response_descending.content))

# Plot the ascending and descending images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ascending_img)
plt.title('Ascending Pass')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(descending_img)
plt.title('Descending Pass')
plt.axis('off')

plt.tight_layout()
plt.show()


