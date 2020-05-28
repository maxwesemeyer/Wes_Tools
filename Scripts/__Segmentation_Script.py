import sys
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
import shutil
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



    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                           crs=gpd.read_file(vector_path).crs)
    #params_bands = [2, 3, 7, 11, 100]
    covs = [1, 2, 5]
    # best set of parameters so far: no PCA, all available bands and Beta=20;
    # according to Liu: no PCA, 11 bands, Beta=100
    #PCA_ = [True, False]
    PCA_ = [False, True]
    params_bands = [3, 7, 11, 17, 21]
    for PC in PCA_:
        for par in params_bands:
            for betas in covs:
                # does not work within function with parallel os.mkdir
                os.mkdir(data_path + 'output')
                #set_global_Cnn_variables(bands=par, convs=betas)

                Parallel(n_jobs=9)(delayed(segment_cnn)(raster_path, vector_geom=row, data_path_output=data_path,
                                                      indexo=index, n_band=par, custom_subsetter=range(3, 62),
                                                        MMU=0.01, convs=betas,
                                                      PCA=PC) for index, row in gdf.iterrows())

                """
                #bayseg
                Parallel(n_jobs=9)(delayed(segment_2)(raster_path, vector_geom=row, data_path_output=data_path,
                                                        indexo=index, n_band=par, custom_subsetter=range(3,62),
                                                      beta_coef=betas, beta_jump=1, MMU=0.01,
                                                      PCA=PC) for index, row in gdf.iterrows())
                """
                joined = join_shapes_gpd(data_path + 'output/', own_segmentation=True)

                if os.path.exists(data_path + 'joined'):
                    print('output directory already exists')

                else:
                    os.mkdir(data_path + 'joined')

                field_counter = "{}{}{}{}{}{}".format(str(PC), "_", str(par), "_", str(betas), '_')
                print(field_counter)
                joined.to_file(data_path + 'joined/joined_cnn' +  field_counter + '.shp')
                shutil.rmtree(data_path + 'output/')

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