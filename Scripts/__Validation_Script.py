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
    vector_path = data_path + 'Vector/Bewrt_paulinaue_3035.gpkg'
    #vector_path = data_path + 'Vector/Paulienenaue_TF.shp'


    list_of_shapes = Shape_finder(data_path + 'joined_bwrt/')
    print(list_of_shapes)
    pse_list = []
    overall_list = []
    clinton_list = []

    for shapes in list_of_shapes:
        #pse = Accuracy_Assessment(vector_path, shapes).IoU()
        pse, nsr, ed2 = Accuracy_Assessment(vector_path, shapes).Liu()
        print(ed2)
        pse_list.append(np.mean(np.array(ed2)))


    print(np.argmin(np.array(pse_list)))

    print(np.array(pse_list), np.argmin(np.array(pse_list)))
    print('According to PSE (ED2)', list_of_shapes[np.argmin(np.array(pse_list))], 'with a score of ', np.min(np.array(pse_list)))
    #plot_shapefile(list_of_shapes[np.argmax(np.array(pse_list))], raster_path)
    #plot_shapefile(list_of_shapes[np.argmax(np.array(pse_list))], raster_path, error_plot=True)
    plot_shapefile(data_path+'Vector/Paulienenaue_TF.shp', raster_path, error_plot=True)
    plot_shapefile(data_path+'Vector/Paulienenaue_TF.shp', raster_path)


    #print('According to Clinton', list_of_shapes[np.argmin(np.array(clinton_list))], 'with a score of ', np.min(np.array(clinton_list)))


"""
data_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/X0068_Y0042/"
    list_raster = Tif_finder(data_path, "^2016.*[S][.][t][i][f]{1,2}$")
    print(list_raster)
    create_stack(list_raster, data_path + 'stacked.tif', n_bands=75 ,custom_subsetter=range(90,165))
"""