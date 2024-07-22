from sentinelsat import SentinelAPI

# Connect to the Copernicus Open Access Hub
api = SentinelAPI('your_username', 'your_password', 'https://scihub.copernicus.eu/dhus')

# Define the date as 14th December 2021
date = "2021-12-14"

# Define the coordinates for the Dammam area
dammam_coordinates = [[[49.82037320425866, 26.485505016295903],
                       [49.82037320425866, 26.040945171190486],
                       [50.23137546499055, 26.040945171190486],
                       [50.23137546499055, 26.485505016295903],
                       [49.82037320425866, 26.485505016295903]]]

# Search for Sentinel-1 data for the Dammam area on the specified date, both ascending and descending orbits
ascending_products = api.query(area=dammam_coordinates,
                               date=(date, date),
                               platformname='Sentinel-1',
                               producttype='GRD',
                               polarisationmode='VV VH',
                               orbitdirection='ASCENDING')

descending_products = api.query(area=dammam_coordinates,
                                date=(date, date),
                                platformname='Sentinel-1',
                                producttype='GRD',
                                polarisationmode='VV VH',
                                orbitdirection='DESCENDING')

# Display metadata for the ascending products
print("Ascending Orbit Products:")
for product_id, product_info in ascending_products.items():
    print("Product ID:", product_id)
    print("Metadata:", product_info)
    print("---------------------------------------")

# Display metadata for the descending products
print("Descending Orbit Products:")
for product_id, product_info in descending_products.items():
    print("Product ID:", product_id)
    print("Metadata:", product_info)
    print("---------------------------------------")