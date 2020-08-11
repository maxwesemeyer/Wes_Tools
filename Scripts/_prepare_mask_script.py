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

    # for each raster in rasterfiles
    # clip alkis mask to raster file
    # write to disk an polygonize
    path = 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN1_BN1/'
    path = 'X:/temp/temp_Max/Data/Train_data/train/label'
    #path = [r'H:\Grassland\X0068_Y0042/2018-2019_180-245_LEVEL4_TSA_SEN2L_BNR_TSI.tif']
    raster_paths = Tif_finder(path, ".*[p][n][g]{1,2}$")
    print(raster_paths)
    #alkis_mask = r'H:\Grassland\X0068_Y0042/GL_mask_2018.tif'
    i = 100

    for raster in raster_paths:
        print(raster)
        splitted = raster.split('/')[-1]
        print(splitted)
        geom = getRasterExtent(raster)
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=0)
            print(out_image.shape, out_image)
            gt_gdal = Affine.to_gdal(out_transform)

            WriteArrayToDisk(out_image.squeeze(), r'X:\SattGruen\Analyse\GLSEG\Raster\snippets_invekos/' + str(splitted), gt_gdal, polygonite=True)
            i += 1

    ####
    # drop duplicate geometries

    vector_paths = Shape_finder(r'X:\SattGruen\Analyse\GLSEG\Raster\snippets_invekos/')
    for vector in vector_paths:
        try:
            df2 = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector)], ignore_index=True),
                             crs="EPSG:3035").drop_duplicates(subset='geometry')
            indexNames = df2[df2['Cluster_nb'] == 0].index
            df2.drop(indexNames, inplace=True)
            df2.to_file(vector)
        except:
            continue



