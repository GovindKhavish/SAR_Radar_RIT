import ee
import pandas as pd
import plotly.express as px

# Initialize the Earth Engine module with specified project
ee.Initialize(project='ee-govindkhavish-rit')

# Define the coordinates of the point (longitude, latitude)
lon = -106.31
lat = 31.97

# Create a point geometry
point = ee.Geometry.Point(lon, lat)

# Define the start and end dates for the time period of interest
start_date = '2023-04-03'
end_date = '2023-12-31'

# Function to filter Sentinel-1 GRD data for VH polarization
def filter_s1_vh(image):
    return image.select('VH').addBands(ee.Image.constant(0).rename('missing'))

# Filter Sentinel-1 GRD data for VH polarization and specified time range
s1_vh_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(point)
                    .filterDate(start_date, end_date)
                    .map(filter_s1_vh))

# Get VH polarization images as a list
vh_image_list = s1_vh_collection.toList(s1_vh_collection.size())

# Initialize lists to store dates and VH polarization intensity
dates = []
vh_intensity = []

# Iterate over the image list and extract dates and VH polarization intensity
for i in range(vh_image_list.size().getInfo()):
    image = ee.Image(vh_image_list.get(i))
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    intensity = image.reduceRegion(ee.Reducer.mean(), point).get('VH')
    # Check if VH intensity is available
    if intensity.getInfo() is not None:
        vh_intensity.append(intensity.getInfo())
    else:
        vh_intensity.append(0)
    dates.append(date)

# Create a DataFrame from the lists
df = pd.DataFrame({'Date': dates, 'VH Intensity': vh_intensity})

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot the interactive chart using plotly
fig = px.line(df, x='Date', y='VH Intensity', title='VH Polarization Intensity over Time')

# Update the layout to make the plot more readable
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='VH Intensity',
    hovermode='x unified',
    template='plotly_dark'
)

# Show the plot
fig.show()






