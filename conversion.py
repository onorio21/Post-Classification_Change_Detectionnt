import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

# Percorsi delle cartelle
input_folder = '/Users/onorio21/Desktop/Università/train'  # Sostituisci con il percorso della cartella "train"
output_folder = '/Users/onorio21/Desktop/Università/conversions'  # Sostituisci con il percorso della cartella di output

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Funzione per processare una singola coppia di cartelle
def process_folder(image_folder, labels_folder, output_subfolder):
    print(f"Processando la cartella: {image_folder} e {labels_folder}")
    
    # Ottieni tutti i file .tif nella cartella image_masked
    tif_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]
    
    if not tif_files:
        print(f"Nessun file .tif trovato nella cartella {image_folder}")
        return
    
    for tif_file in tif_files:
        # Percorsi completi dei file .tif e .geojson
        rgb_image_path = os.path.join(image_folder, tif_file)
        geojson_file = tif_file.replace('.tif', '_Buildings.geojson')
        geojson_path = os.path.join(labels_folder, geojson_file)

        # Verifica che il file .geojson corrispondente esista
        if not os.path.exists(geojson_path):
            print(f"File GeoJSON non trovato per {tif_file}, saltato.")
            continue

        print(f"Processando il file: {rgb_image_path} con {geojson_path}")

        # Carica l'immagine RGB per ottenere le dimensioni e la trasformazione affine
        with rasterio.open(rgb_image_path) as src:
            meta = src.meta.copy()
            transform = src.transform
            width = src.width
            height = src.height

        # Carica il file GeoJSON
        gdf = gpd.read_file(geojson_path)

        # Verifica che il CRS del GeoJSON corrisponda a quello dell'immagine
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Debugging: Verifica delle geometrie nel GeoJSON
        if gdf.empty or gdf.geometry.is_empty.all():
            print(f"Nessuna geometria valida trovata in {geojson_path}, saltato.")
            continue

        # Rasterizza le geometrie nel GeoDataFrame
        shapes = ((geom, 1) for geom in gdf.geometry if not geom.is_empty)
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
        output_raster_path = os.path.join(output_subfolder, tif_file)

        # Scrivi l'immagine binaria sul disco senza compressione
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(binary_raster, 1)

        print(f"Immagine binaria salvata in: {output_raster_path}")

# Scansiona tutte le sottocartelle nella cartella di input
for dir_name in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, dir_name)
    
    # Percorso delle cartelle "images_masked" e "labels_match"
    image_folder = os.path.join(subfolder_path, 'images_masked')
    labels_folder = os.path.join(subfolder_path, 'labels_match')

    if os.path.exists(image_folder) and os.path.exists(labels_folder):
        print(f"Trovate cartelle 'images_masked' e 'labels_match' in: {subfolder_path}")
        
        # Costruisci il percorso della cartella di output corrispondente
        output_subfolder = os.path.join(output_folder, dir_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Processa i file nella cartella corrente
        process_folder(image_folder, labels_folder, output_subfolder)
    else:
        print(f"Le cartelle 'images_masked' e 'labels_match' non sono presenti in: {subfolder_path}")

print("Conversione completata.")