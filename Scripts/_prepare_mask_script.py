import sys
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
import shutil
import gdal
sys.path.append("X:/temp/temp_Max/")
import rasterio
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *
from affine import Affine

if __name__ == '__main__':
    """
    # for each raster in rasterfiles
    # clip alkis mask to raster file
    # write to disk an polygonize
    path = 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN1_BN1/'

    raster_paths = Tif_finder(path, ".*[N][D][V].*[S][S].*[t][i][f]{1,2}$")
    alkis_mask = r'X:\SattGruen\Analyse\GLSEG\Raster\ALKIS_GL_Maske_2019_3035.tif'
    i = 1
    for raster in raster_paths:
        print(raster)
        geom = getRasterExtent(raster)
        with rasterio.open(alkis_mask) as src:
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=0)
            print(out_image.shape, out_image)
            gt_gdal = Affine.to_gdal(out_transform)

            WriteArrayToDisk(out_image.squeeze(), r'X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/' + str(i), gt_gdal, polygonite=True)
            i += 1
    """
    ####
    # drop duplicate geometries
    vector_paths = Shape_finder('X:/SattGruen/Analyse\GLSEG/Raster/Vectorized_Alkis/')
    for vector in vector_paths:

        df2 = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector)], ignore_index=True),
                         crs="EPSG:3035").drop_duplicates(subset='geometry')

        df2.to_file(vector)


