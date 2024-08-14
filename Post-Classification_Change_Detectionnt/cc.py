import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Percorsi dei file
rgb_image_path = '/Users/onorio21/Desktop/Università/train/L15-0331E-1257N_1327_3160_13/images/global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif'
geojson_path = '/Users/onorio21/Desktop/Università/train/L15-0331E-1257N_1327_3160_13/labels_match/global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13_Buildings.geojson'
output_raster_path = '/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/provaraster/output.tif'

# Carica l'immagine RGB per ottenere le dimensioni e la trasformazione affine
with rasterio.open(rgb_image_path) as src:
    meta = src.meta.copy()
    transform = src.transform
    width = src.width
    height = src.height
    bounds = src.bounds

# Carica il file GeoJSON
gdf = gpd.read_file(geojson_path)

# Verifica che il CRS del GeoJSON corrisponda a quello dell'immagine
if gdf.crs != src.crs:
    gdf = gdf.to_crs(src.crs)

# Rasterizza le geometrie nel GeoDataFrame
shapes = ((geom, 1) for geom in gdf.geometry)
binary_raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')

binary_raster = (binary_raster * 255).astype(np.uint8)

# Debug: mostrare un'anteprima dell'immagine binaria
plt.imshow(binary_raster, cmap='gray')
plt.title("Anteprima immagine binaria")
plt.show()

# Assicurati che 'nodata' sia None e non usare compressione
meta.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": 'uint8',
    "nodata": None,
    "compress": None  # Rimuovi la compressione
})

# Scrivi l'immagine binaria sul disco senza compressione
with rasterio.open(output_raster_path, 'w', **meta) as dst:
    dst.write(binary_raster, 1)

# Debug: rileggi il file appena scritto e visualizza l'immagine
with rasterio.open(output_raster_path) as src:
    reloaded_raster = src.read(1)
    plt.imshow(reloaded_raster, cmap='gray')
    plt.title("Immagine binaria ricaricata dal file salvato")
    plt.show()

print(f"Immagine binaria salvata in: {output_raster_path}")