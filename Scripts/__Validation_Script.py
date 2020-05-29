import sys
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
import shutil
import glob
sys.path.append("X:/temp/temp_Max/")

from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.create_vrt import *

if __name__ == '__main__':




    data_path = 'X:/temp/temp_Max/Data/'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster'
    raster_path = 'X:/SattGruen/Analyse/GLSEG/Raster/X0068_Y0042/2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_TSS.tif'
    vector_path = data_path + 'Vector/dissolved_paulinaue_3035_parcels.gpkg'

    list_of_shapes = Shape_finder(data_path + 'joined/')
    print(list_of_shapes)
    pse_list = []
    overall_list = []
    clinton_list = []

    for shapes in list_of_shapes:

        try:
            pse, nsr, ed2 = Accuracy_Assessment.Liu(data_path + 'Vector/Paulienenaue_TF.shp', shapes)
            print(np.mean(np.array(ed2)))
            US, OS, Overall = Accuracy_Assessment.Clinton(data_path + 'Vector/Paulienenaue_TF.shp', shapes)
            print(np.mean(np.array(Overall)))
            overall_list.append(np.mean(np.array(ed2)))
            pse_list.append(np.mean(np.array(pse)))
            clinton_list.append(np.mean(np.array(Overall)))
        except:
            pse_list.append(10)
            overall_list.append(10)
            print('for some reason not working')

    print(np.argmin(np.array(pse_list)))
    print(np.argmin(np.array(overall_list)))
    print(np.array(pse_list), np.argmin(np.array(pse_list)), np.argmin(np.array(overall_list)))
    print('According to PSE (ED2)', list_of_shapes[np.argmin(np.array(overall_list))], 'with a score of ', np.min(np.array(overall_list)))
    print('According to Clinton', list_of_shapes[np.argmin(np.array(clinton_list))], 'with a score of ', np.min(np.array(clinton_list)))


"""
data_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/X0068_Y0042/"
    list_raster = Tif_finder(data_path, "^2016.*[S][.][t][i][f]{1,2}$")
    print(list_raster)
    create_stack(list_raster, data_path + 'stacked.tif', n_bands=75 ,custom_subsetter=range(90,165))
"""