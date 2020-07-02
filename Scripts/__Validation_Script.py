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
    raster_path = 'X:/SattGruen/Analyse/GLSEG/Raster/X0068_Y0042/2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_TSI.tif'
    vector_path = data_path + 'Vector/Bewrt_paulinaue_3035.gpkg'
    #vector_path = data_path + 'Vector/Paulienenaue_TF.shp'

    #plot_shapefile(data_path + 'Vector/result_.gpkg', raster_path, error_plot=True, trample_check=False, custom_subsetter=range(1, 60))

    list_of_shapes = Shape_finder(data_path + 'joined/')
    print(list_of_shapes)
    pse_list = []
    iou_list = []
    overall_list = []
    clinton_list = []

    for shapes in list_of_shapes:
        #pse = Accuracy_Assessment(vector_path, shapes).IoU()
        pse, nsr, ed2 = Accuracy_Assessment(vector_path, shapes, convert_reference=True, raster=raster_path).Liu()
        OS, US, Overall = Accuracy_Assessment(vector_path, shapes, convert_reference=True, raster=raster_path).Clinton()
        iou = Accuracy_Assessment(vector_path, shapes, convert_reference=True, raster=raster_path).IoU()
        print((np.array(iou)))
        print(shapes, np.mean(np.array(OS)), np.mean(np.array(US)), np.mean(np.array(Overall)))
        pse_list.append(np.mean(np.array(ed2)))
        iou_list.append(np.mean(np.array(iou)))
    dict = {'name': list_of_shapes, 'IoU': iou_list, 'Clinton': pse_list}
    score_frame = pd.DataFrame(dict)
    score_frame.to_csv(data_path + 'scores.csv')
    print(score_frame)
    #print(np.argmax(np.array(pse_list)), list_of_shapes, pse_list)

    #print(np.array(pse_list), np.argmin(np.array(pse_list)))
    print('According to PSE (ED2)', list_of_shapes[np.argmin(np.array(pse_list))], 'with a score of ',
          np.min(np.array(pse_list)))
    print('According to IoU', list_of_shapes[np.argmax(np.array(iou_list))], 'with a score of ',
          np.max(np.array(iou_list)))
    plot_shapefile(list_of_shapes[np.argmin(np.array(pse_list))], raster_path)
    plot_shapefile(list_of_shapes[np.argmin(np.array(pse_list))], raster_path, error_plot=True)
    plot_shapefile(data_path+'Vector/Paulienenaue_TF.shp', raster_path, error_plot=True)
    plot_shapefile(data_path+'Vector/Paulienenaue_TF.shp', raster_path)


    #print('According to Clinton', list_of_shapes[np.argmin(np.array(clinton_list))], 'with a score of ', np.min(np.array(clinton_list)))


