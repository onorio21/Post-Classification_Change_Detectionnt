import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio as rio
from rasterio import features
from pathlib import Path
import pathlib
import geopandas as gpd
from descartes import PolygonPatch
from PIL import Image
import itertools
import re

def plot_gdf(gdf,fill=False,ax=None,linewidth=0.2):
    if ax is None:
        _,ax = plt.subplots(1,figsize=(3, 3))
        
    for geom in gdf['geometry']:
        if geom.is_empty:  # Skip empty geometries
            continue
        elif geom.type in ['Polygon', 'MultiPolygon']:
            if fill:
                patch = PolygonPatch(geom,linewidth=linewidth,color='fuchsia')
                ax.add_patch(patch)
            else:
                ax.plot(*geom.exterior.xy,linewidth=linewidth)

        else:
            print(f"Skipping geometry of type {geom.type}, not supported for PolygonPatch")  # Informative message
    return(ax)


def plot_sat(path,gdf=None, fill=False,linewidth=0.2):
    f, ax = plt.subplots(1,figsize=(3, 3))
    f.tight_layout()
    
    r = rio.open(path)
    r = r.read()
    r = r.transpose((1,2,0,))
    ax.imshow(r)
    
    if gdf is not None:
        plot_gdf(gdf,fill=fill,ax=ax,linewidth=linewidth)
        
    return(ax)


test_raster_path = '/Users/onorio21/Desktop/Università/analisi/L15-1716E-1211N_6864_3345_13/global_monthly_2018_01_mosaic_L15-1716E-1211N_6864_3345_13.tif'
test_raster_path_24 = '/Users/onorio21/Desktop/Università/analisi/L15-1716E-1211N_6864_3345_13/global_monthly_2020_01_mosaic_L15-1716E-1211N_6864_3345_13.tif'
test_geojson_path = '/Users/onorio21/Desktop/Università/train/L15-1716E-1211N_6864_3345_13/labels_match_pix/global_monthly_2018_01_mosaic_L15-1716E-1211N_6864_3345_13_Buildings.geojson'
test_geojson_path_24 = '/Users/onorio21/Desktop/Università/train/L15-1716E-1211N_6864_3345_13/labels_match_pix/global_monthly_2020_01_mosaic_L15-1716E-1211N_6864_3345_13_Buildings.geojson'

test_gdf = gpd.read_file(test_geojson_path)
test_gdf_24 = gpd.read_file(test_geojson_path_24)

test_gdf.set_index('Id',inplace=True)
test_gdf_24.set_index('Id',inplace=True)

test_gdf.sort_index(inplace=True)
test_gdf_24.sort_index(inplace=True)


plot_gdf(test_gdf)
plt.show()
plot_sat(path=test_raster_path)
plt.show()


plot_sat(path=test_raster_path,gdf=test_gdf)
plt.show()

plot_sat(path=test_raster_path_24,gdf=test_gdf_24)
plt.show()