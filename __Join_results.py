import os
import geopandas as gpd
import pandas as pd
import numpy as np


def join_shapes_gpd(path_to_shapes):
    file = os.listdir(path_to_shapes)
    path = [os.path.join(path_to_shapes, i) for i in file if ".shp" in i]
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path], ignore_index=True), crs=gpd.read_file(path[0]).crs)

    # drop cluster number 0, which is all no grassland polygons
    indexNames = gdf[gdf['Cluster_nb'] == 0].index
    gdf.drop(indexNames, inplace=True)

    # drop all entries with field nb = na, which don't have a geometry and are duplicates
    indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
    gdf.drop(indexNames_2, inplace=True)
    return gdf
