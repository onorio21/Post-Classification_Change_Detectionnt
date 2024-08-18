import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load the first binary image
with rasterio.open('/Users/onorio21/Desktop/Università/conversions/L15-0331E-1257N_1327_3160_13/global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif') as src1:
    image1 = src1.read(1)  # Read the first band

# Load the second binary image
with rasterio.open('/Users/onorio21/Desktop/Università/conversions/L15-0331E-1257N_1327_3160_13/global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13.tif') as src2:
    image2 = src2.read(1)  # Read the first band

if image1.shape != image2.shape:
    raise ValueError("The images do not have the same dimensions.")

# Compute the difference between the two images
change = image1 - image2

# Convert the difference to a binary change map
# 1 indicates a change, 0 indicates no change
change_map = np.where(change != 0, 1, 0)

# Define the output file path
output_file = 'change_map.tif'

# Copy the metadata from one of the input images
meta = src1.meta.copy()

# Update the metadata for the output image
meta.update(dtype=rasterio.uint8, count=1)

change_map = (change_map * 255).astype(np.uint8)

# Write the change map to a new GeoTIFF file
with rasterio.open(output_file, 'w', **meta) as dst:
    dst.write(change_map.astype(rasterio.uint8), 1)

plt.imshow(change_map, cmap='gray')
plt.show()