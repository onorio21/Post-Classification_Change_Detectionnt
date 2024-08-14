import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

# Percorsi delle cartelle
tif_folder = '/Users/onorio21/Desktop/Università/train/L15-0358E-1220N_1433_3310_13/images_masked'  # Sostituisci con il percorso della cartella contenente i file .tif
geojson_folder = '/Users/onorio21/Desktop/Università/train/L15-0358E-1220N_1433_3310_13/labels_match'  # Sostituisci con il percorso della cartella contenente i file .geojson
output_folder = '/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/provaraster/L15-0358E-1220N_1433_3310_13'  # Sostituisci con il percorso della cartella di output

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Ottieni tutti i file .tif nella cartella specificata
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

for tif_file in tif_files:
    # Percorsi completi dei file .tif e .geojson
    rgb_image_path = os.path.join(tif_folder, tif_file)
    geojson_file = tif_file.replace('.tif', '_Buildings.geojson')
    geojson_path = os.path.join(geojson_folder, geojson_file)

    # Verifica che il file .geojson corrispondente esista
    if not os.path.exists(geojson_path):
        print(f"File GeoJSON non trovato per {tif_file}, saltato.")
        continue

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

    # Aggiorna i metadati per il file di output
    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": 'uint8',
        "nodata": None,
        "compress": None  # Rimuovi la compressione
    })

    # Percorso del file di output
    output_raster_path = os.path.join(output_folder, tif_file)

    # Scrivi l'immagine binaria sul disco senza compressione
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(binary_raster, 1)

    print(f"Immagine binaria salvata in: {output_raster_path}")

print("Conversione completata.")
