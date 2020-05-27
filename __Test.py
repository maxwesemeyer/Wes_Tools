import sys
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
sys.path.append("X:/temp/temp_Max/")

from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *


if __name__ == '__main__':
    data_path = 'X:/temp/temp_Max/Data/'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster'
    raster_path = 'X:/SattGruen/Analyse/GLSEG/Raster/X0068_Y0042/2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_TSS.tif'
    vector_path = data_path + 'Vector/dissolved_paulinaue_3035_parcels.gpkg'

    set_global_Cnn_variables(bands=7, convs=3)

    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                           crs=gpd.read_file(vector_path).crs)
    params = [1,2,3]
    for par in params:
        """
        Parallel(n_jobs=1)(delayed(segment_cnn)(raster_path, vector_geom=row, data_path_output=data_path,
                                                indexo=index, n_band=7, custom_subsetter=range(5,60),
                                                                 MMU=0.01, PCA=False) for index, row in gdf.iterrows())
        """
        joined = join_shapes_gpd(data_path + 'output/', own_segmentation=True)
        joined.to_file(data_path + 'output/joined.gpkg')

        US_out, OS_out, Overall_out = Accuracy_Assessment.Liu(data_path + 'Vector/Paulienenaue_TF.shp', data_path + 'output/joined.gpkg')
        print(np.mean(np.array(US_out)), np.mean(np.array(OS_out)), np.mean(np.array(Overall_out)))
        US_out, OS_out, Overall_out = Accuracy_Assessment.Clinton(data_path + 'Vector/Paulienenaue_TF.shp', data_path + 'output/joined.gpkg')
        print(np.mean(np.array(US_out)), np.mean(np.array(OS_out)), np.mean(np.array(Overall_out)))
        os.remove(data_path + 'output/joined.gpkg')
    """
    # drop cluster number 0, which is all no grassland polygons
    indexNames = gdf[gdf['Cluster_nb'] == 0].index
    gdf.drop(indexNames, inplace=True)

    # drop all entries with field nb = na, which don't have a geometry and are duplicates
    indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
    gdf.drop(indexNames_2, inplace=True)

    x = Parallel(n_jobs=1)(
        delayed(aggregator)(
            raster_NDV='X:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif',
            shapei=row, indexo=index, subsetter=None) for
        index, row in gdf.iterrows())
    """