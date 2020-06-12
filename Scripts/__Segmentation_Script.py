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
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/Paulinenaue/X0068_Y0042/stacked.tif'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/S-1/X0068_Y0042/X0068_Y0042_stacked.tif'

    # 2016-2019_001-365_LEVEL4_TSA_LNDLG_GRN_TSS
    #raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/S-1/vrt/vrt_global.vrt"
    #raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/X0068_Y0042S1_S2_stack.tif"
    raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/vrt/vrt_global.vrt"
    #vector_path = data_path + 'Vector/paulinaue_bwrt_diss_parcels_3035.shp'
    vector_path = data_path + 'Vector/dissolved_paulinaue_3035_parcels.gpkg'
    #vector_path = data_path + 'Vector/Ribbeck_grassland_LAEA_europe.shp'

    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                           crs=gpd.read_file(vector_path).crs)
    #params_bands = [2, 3, 7, 11, 100]
    covs = [1]
    # best set of parameters so far: no PCA, all available bands and Beta=20;
    # according to Liu: no PCA, 11 bands, Beta=100
    #PCA_ = [True, False]
    PCA_ = [False]
    params_bands = [10, 15]
    #params_bands = [2, 3, 5, 7]
    for PC in PCA_:
        for par in params_bands:
            for betas in covs:
                # does not work within function with parallel os.mkdir
                os.mkdir(data_path + 'output')
                #set_global_Cnn_variables(bands=par, convs=betas)
                # old subsetter range(1,500)
                # 104 168 = April - Ende Oktober
                Parallel(n_jobs=3)(delayed(segment_2)(data_patg_alt, vector_geom=row, data_path_output=data_path,
                                                      indexo=index, n_band=par, custom_subsetter=range(1, 64), #custom_subsetter=range(1, 300),# #custom_subsetter=range(1,392),
                                                        MMU=0.01, beta_coef=80, beta_jump=1,
                                                      PCA=PC) for index, row in gdf.iterrows())

               
                joined = join_shapes_gpd(data_path + 'output/', own_segmentation='own')

                if os.path.exists(data_path + 'joined'):
                    print('output directory already exists')

                else:
                    os.mkdir(data_path + 'joined')

                field_counter = "{}{}{}{}{}{}".format(str(PC), "_", str(par), "_", str(betas), '_')
                print(field_counter)
                joined.to_file(data_path + 'joined/bayseg_S1_' +  field_counter + '.shp')
                shutil.rmtree(data_path + 'output/')
    

"""
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file('X:/temp/temp_Max/Data/joined_bwrt//bayseg_bwrtFalse_20_1_.shp')], ignore_index=True),
                           crs=gpd.read_file('X:/temp/temp_Max/Data/joined_bwrt//bayseg_bwrtFalse_20_1_.shp').crs)
    # drop cluster number 0, which is all no grassland polygons
    indexNames = gdf[gdf['Cluster_nb'] == 0].index
    gdf.drop(indexNames, inplace=True)

    # drop all entries with field nb = na, which don't have a geometry and are duplicates
    indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
    gdf.drop(indexNames_2, inplace=True)

    x = Parallel(n_jobs=1)(
        delayed(aggregator)(
            raster_NDV=data_patg_alt,
            shapei=row, indexo=index, subsetter=range(90,165)) for
        index, row in gdf.iterrows())

    mergo = pd.DataFrame(x)
    print(list(mergo.columns.values))

    mergo[mergo.columns[-1]] = mergo[mergo.columns[-1]].astype(dtype=int)
    merged = gdf.merge(mergo, left_index=True, right_index=False, right_on=mergo[mergo.columns[-1]])
    merged = merged.iloc[:,range(3, 151)]
    #gpd_merged = gpd.GeoDataFrame(merged, crs="EPSG:3035", geometry=merged[0])
    #gpd_merged.to_file(data_path + 'merged_bayseg_raster.shp')
    merged.to_csv(data_path + 'merged_bayseg_raster.csv')
"""

"""
IDEAs: 

define/find areas of no/low regrowth; the VI values in these areas can then be adapted
first idea: find points below a NDVI value of ~0.5 then see if regrowth or not
unsupervised video segmentation? 

TODO:
Sentinel 1 für Paulinaue; Spectral Temporal metrics für paulinaue; Mit Grünlandmaske für Paulinaue
Für mowing detection: gesamtes gebiet mit spectral temproal metrics; Coherence; combined; PCA

"""