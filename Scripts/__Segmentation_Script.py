import sys
from shapely.geometry import Polygon
import shapely
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
import shutil
import gdal
sys.path.append("X:/temp/temp_Max/")
import glob
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *


if __name__ == '__main__':

    data_path = 'X:/temp/temp_Max/Data/'
    #data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/Paulinenaue/X0068_Y0042/S1_S2_stack.tif'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/Ramin/X0071_Y0039/stacked.tif'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/NDVI_MV/X0071_Y0039/2016-2019_001-365_LEVEL4_TSA_SEN2L_NDV_C0_S0_FAVG_TY_C95T_TSS.dat'


    #mask = gdal.Open('X:/SattGruen/Analyse/GLSEG/Raster/ALKIS_GL_Maske_2019_3035.tif')
    #mask_arr = mask.ReadAsArray()
    #WriteArrayToDisk(mask_arr, 'X:/SattGruen/Analyse/GLSEG/Raster/ALKIS_GL_Maske_2019_3035_del',
    #                 mask.GetGeoTransform(), polygonite=True, fieldo=None, EPSG=3035)

    #data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster/S-1/X0068_Y0042/X0068_Y0042_stacked.tif'

    # 2016-2019_001-365_LEVEL4_TSA_LNDLG_GRN_TSS
    #raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/S-1/vrt/vrt_global.vrt"
    #raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/X0068_Y0042S1_S2_stack.tif"
    raster_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/vrt/vrt_global.vrt"
    vector_path = data_path + 'Vector/dissolved_paulinaue_3035_parcels.gpkg'
    vector_path = data_path + 'Vector/MVP_subset_71_39_3035.gpkg'
    #vector_path = data_path + 'Vector/Dissolved_all_bewrt_3035.shp'
    #vector_path = data_path + 'Vector/Ribbeck_grassland_LAEA_europe.shp'


    vector_paths = glob.glob(r'X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/' + '*.shp')
    vector_paths = ['X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/12polygonized.shp']
    another_counter = 0
    for vector_path in vector_paths:
        data_patg_alt = find_matching_raster(vector_path, 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/', ".*[E][V][I].*[B][M].*[t][i][f]{1,2}$")
        if data_patg_alt is None:
            continue
        big_box = getRasterExtent(data_patg_alt)
        boxes = fishnet(big_box, 10000)
        gdf_ = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                               crs="EPSG:3035")
        mask = gdf_.area > 2500
        gdf = gdf_.loc[mask]

        indexNames = gdf[gdf['Cluster_nb'] == 0].index
        gdf.drop(indexNames, inplace=True)

        box_counter = 100
        for poly in boxes:
            sub = gpd.GeoDataFrame(gpd.clip(gdf.buffer(0), Polygon(poly).buffer(0.001)))
            #sub = gdf[gdf.geometry.intersection(poly)]
            print(sub)
            #params_bands = [2, 3, 7, 11, 100]
            covs = [30]
            # best set of parameters so far: no PCA, all available bands and Beta=20;
            # according to Liu: no PCA, 11 bands, Beta=100
            #PCA_ = [True, False]
            PCA_ = [True]
            #params_bands = [10, 20, 25]
            params_bands = [2]
            for PC in PCA_:
                for par in params_bands:
                    for betas in covs:
                        # does not work within function with parallel os.mkdir
                        os.mkdir(data_path + 'output')
                        #set_global_Cnn_variables(bands=par, convs=betas)
                        # old subsetter range(1,500)
                        # 104 168 = April - Ende Oktober
                        Parallel(n_jobs=3)(delayed(segment_2)(data_patg_alt, vector_geom=row, data_path_output=data_path,
                                                              indexo=index, n_band=par, custom_subsetter=range(1, 12),# custom_subsetter=range(1, 4*12), #custom_subsetter=range(1, 300),# #custom_subsetter=range(1,392),
                                                                MMU=0.1,into_pca=betas, beta_coef=80, beta_jump=1,
                                                              PCA=PC) for index, row in sub.iterrows())

                        if not os.listdir(data_path + 'output/'):
                            print('directory empty')
                        else:
                            joined = join_shapes_gpd(data_path + 'output/', own_segmentation='own')

                        if os.path.exists(data_path + 'joined'):
                            print('output directory already exists')

                        else:
                            os.mkdir(data_path + 'joined')

                        field_counter = "{}{}{}{}{}{}{}{}".format(str(PC), "_", str(par), "_", str(betas), '_', box_counter, "_" + str(another_counter) + "_")
                        box_counter += 1
                        another_counter += 10
                        print(field_counter)
                        joined.to_file(data_path + 'joined/mowing' +  field_counter + '.shp')
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