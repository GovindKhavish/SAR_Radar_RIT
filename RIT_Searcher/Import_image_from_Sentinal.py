import ee
import geemap

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

# Function to convert radar data to decibels
def toDB(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

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
ascending_vv = ascending_collection.select('VV').map(toDB)
ascending_vh = ascending_collection.select('VH').map(toDB)

# Select the polarization bands and convert to decibels for descending passes
descending_vv = descending_collection.select('VV').map(toDB)
descending_vh = descending_collection.select('VH').map(toDB)

# Merge the ascending VV and VH collections
ascending_image = ascending_vv.map(lambda img: img.addBands(ascending_vh.filterDate(img.date()).first()))

# Merge the descending VV and VH collections
descending_image = descending_vv.map(lambda img: img.addBands(descending_vh.filterDate(img.date()).first()))

# Mosaic the merged collections
ascending_image = ascending_image.mosaic()
descending_image = descending_image.mosaic()

# Create an interactive map
Map = geemap.Map(center=[26.2639, 50.2083], zoom=10)

# Add the ascending and descending images to the map
Map.addLayer(ascending_image, {'bands': ['VV', 'VH', 'VV'], 'min': -25, 'max': 5}, 'Ascending Pass')
Map.addLayer(descending_image, {'bands': ['VV', 'VH', 'VV'], 'min': -25, 'max': 5}, 'Descending Pass')

# Display the map
Map.show()
