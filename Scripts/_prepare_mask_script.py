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
    #path = [r'H:\Grassland\X0068_Y0042/2018-2019_180-245_LEVEL4_TSA_SEN2L_BNR_TSI.tif']
    raster_paths = Tif_finder(path, ".*[t][i][f]{1,2}$")
    print(raster_paths)
    #alkis_mask = r'H:\Grassland\X0068_Y0042/GL_mask_2018.tif'
    i = 100
    """
    for raster in raster_paths:
        print(raster)
        splitted = raster.split('/')
        filename = splitted[-1]
        force = splitted[-2]
        print(splitted)

        raster_src = r'H:\Grassland\EVI/' + force + '/GL_mask_2018.tif'
        print(raster_src)
        geom = getRasterExtent(raster_src)
        with rasterio.open(raster_src) as src:
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=0)
            print(out_image.shape, out_image)
            gt_gdal = Affine.to_gdal(out_transform)
            out_path = r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\AN3_BN1/' + str(force) + '/' + 'Gl_mask_2018'
            print(out_path)
            WriteArrayToDisk(out_image.squeeze(), out_path, gt_gdal, polygonite=True)
            i += 1
    
    ####
    # drop duplicate geometries

    vector_paths = Shape_finder(r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\AN3_BN1/')
    print(vector_paths)
    for vector in vector_paths:
        try:
            df2 = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector)], ignore_index=True),
                             crs="EPSG:3035").drop_duplicates(subset='geometry')
            indexNames = df2[df2['Cluster_nb'] == 0].index
            df2.drop(indexNames, inplace=True)
            df2 = df2.explode()
            df2.to_file(vector)
        except:
            continue
"""
    ##### i
    adapt_to_raods = True

    gdf_roads = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(r'X:\Data\Vector\OSM\4mw\Roads_small/germany-highway-buffer-epsg3035.shp')],
                  ignore_index=True),
        crs="EPSG:3035")
    #gdf_roads['geometry'] = gdf_roads.unary_union
    #gdf_roads.to_file(r'X:\temp\temp_Max\Qgis/lines_unarunion.gpkg')
    if adapt_to_raods:
        import glob
        dp_data = 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/'
        vector_paths = glob.glob(dp_data + '*.shp', recursive=True)
        vector_paths = Shape_finder(dp_data)
        vector_paths = vector_paths[3:]
        print(vector_paths)
        another_counter = 0

        for vector_path in vector_paths:
            force_tile = vector_path.split('/')[-2]
            print(vector_path, force_tile)
            gdf_ = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                                    crs="EPSG:3035")
            gdf_ = gpd.overlay(gdf_, gdf_roads, how='difference')
            gdf_.to_file(vector_path)


