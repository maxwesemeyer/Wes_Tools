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

    list_of_shapes = Shape_finder('W:/Student_Data/Wesemeyer/Master/results_new/')
    print(list_of_shapes)
    pse_list = []
    overall_list = []

    for shapes in list_of_shapes:

        try:
            gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(list_of_shapes[23])], ignore_index=True),
                                   crs=gpd.read_file(list_of_shapes[23]).crs)

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
            mergo = pd.DataFrame(x)
            print('Mergo:', mergo)
        except:
            print('error')
            continue

"""
        try:
            DistancoTron('W:/Student_Data/Wesemeyer/Projekt_/Shapes/RH_UG_3035.shp',
                         'X:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif')
            pse, nsr, ed2 = Accuracy_Assessment.Liu('W:/Student_Data/Wesemeyer/Projekt_/Shapes/RH_UG_3035.shp', shapes)
            #print(np.mean(np.array(ed2)))
            US, OS, Overall = Accuracy_Assessment.Clinton('W:/Student_Data/Wesemeyer/Projekt_/Shapes/RH_UG_3035.shp', shapes)
            #print(np.mean(np.array(Overall)))
            pse_list.append(np.mean(np.array(ed2)))
            overall_list.append(np.mean(np.array(Overall)))
        except:
            pse_list.append(10)
            overall_list.append(10)
            print('for some reason not working')

    print(np.argmin(np.array(pse_list)))
    print(np.argmin(np.array(overall_list)))
    print(np.array(pse_list), np.argmin(np.array(pse_list)), np.argmin(np.array(overall_list)))
    print('According to PSE (ED2)', list_of_shapes[np.argmin(np.array(pse_list))], 'with a score of ', np.min(np.array(pse_list)))
    print('According to Clinton', list_of_shapes[np.argmin(np.array(overall_list))], 'with a score of ', np.min(np.array(overall_list)))
