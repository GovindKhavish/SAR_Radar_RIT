# Basic test program to view a single a GRD image from Sentenial-1 in VsCode using python and Google Earth Engine
# Version 0.1
# Modified: 2024/05/25

# Imports
import ee # Earth Engine API
import numpy as np # Numerical operations and array manipluation
import matplotlib.pyplot as plt # Plotting anf visualization
from PIL import Image # Python Pillow library for images
import urllib.request # Opening URLs

# Initialize the Earth Engine module
ee.Initialize(project='ee-govindkhavish-rit') # Specifying the project linked to my account to access data

# Define the coordinates for the Dammam area given in the Bellingcat code
dammam_coordinates = [[49.82037320425866, 26.485505016295903],
                      [49.82037320425866, 26.040945171190486],
                      [50.23137546499055, 26.040945171190486],
                      [50.23137546499055, 26.485505016295903],
                      [49.82037320425866, 26.485505016295903]]

# Create an Earth Engine geometry object for the Dammam area
dammam_geometry = ee.Geometry.Polygon(dammam_coordinates)

# Collects the sentinal data using a set of filters based on insturment mode, polarization, oribit properties and polarizaition band
collection_VV = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                 .select(['VV']))

# Create date range to be looked at
start_date = '2021-12-01'
end_date = '2021-12-31'

# Creates a mean of all the images collected in the date range
composite = collection_VV.filterDate(start_date, end_date).mean()

# Get the image URL with the specified region ~ Earth Engine gives back the url linked to the image/data you requested
# Cliping functions cuts the excess image out just tot he co-ordinates speicfied
# Min and max are visualization parameters which help enahnce the differences
url = composite.clip(dammam_geometry).getThumbURL({'min': -25, 'max': 0, 'region': dammam_geometry, 'dimensions': 512})

# Open the image from the URL using Pillow and convert it to a NumPy array
image = Image.open(urllib.request.urlopen(url))

# Convert the image to grayscale
# The image request are in form two chanels per pixel instead of RGB normal 3 chanels per pixel
# Current form is (512, 474, 2) but goes into a (512, 474)
image_gray = image.convert('L')

# Convert the grayscale image to a NumPy array
# Python can handle turning a single channel into a 3 chanel RGB image is much easier
image_array = np.array(image_gray)

# Display the image
plt.imshow(image_array, cmap='viridis')
plt.axis('off')
plt.show()

