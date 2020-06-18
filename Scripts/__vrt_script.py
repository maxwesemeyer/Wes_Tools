import glob
import gdal
import os
import re
import sys
sys.path.append("X:/temp/temp_Max/")

from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.create_vrt import *

if __name__ == '__main__':

    data_path_input = "X:/SattGruen/Analyse/GLSEG/Raster/Ramin/X0071_Y0039/"
    file_path_raster = Tif_finder(data_path_input)
    print(file_path_raster)
    create_stack(file_path_raster, data_path_input + 'stacked.tif', n_bands=12, custom_subsetter=range(1, 12))
    print('finished')


    folders_BRB = {"X0065_Y0040", "X0065_Y0041", "X0066_Y0040", "X0066_Y0041", "X0066_Y0042", "X0067_Y0040", "X0067_Y0041",
                   "X0067_Y0042", "X0067_Y0043", "X0067_Y0044", "X0067_Y0045", "X0068_Y0040", "X0068_Y0041", "X0068_Y0042",
                   "X0068_Y0043", "X0068_Y0044", "X0068_Y0045", "X0069_Y0040", "X0069_Y0041", "X0069_Y0042", "X0069_Y0043",
                   "X0069_Y0044", "X0069_Y0045", "X0069_Y0046", "X0069_Y0047", "X0070_Y0039", "X0070_Y0040", "X0070_Y0041",
                   "X0070_Y0042", "X0070_Y0043", "X0070_Y0044", "X0070_Y0045", "X0070_Y0046", "X0070_Y0047", "X0071_Y0039",
                   "X0071_Y0040", "X0071_Y0041", "X0071_Y0042", "X0071_Y0043", "X0071_Y0044", "X0071_Y0045", "X0071_Y0046",
                   "X0071_Y0047", "X0072_Y0040", "X0072_Y0042", "X0072_Y0043", "X0072_Y0044", "X0072_Y0045", "X0072_Y0046",
                   "X0072_Y0047", "X0073_Y0044", "X0073_Y0045", "X0073_Y0046"}
    """
    data_path_input = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/"
    file_path_raster = Tif_finder(data_path_input, "^2016.*[S][.][t][i][f]{1,2}$")
    print(file_path_raster)
    data_path_vrt = data_path_input + 'vrt/'
    if not os.path.exists(data_path_vrt):
        os.mkdir(data_path_vrt)

    topo = False
    soil = False

    temp_clim = False
    prec_clim = False

    temp_mete = False
    prec_mete = False
    smoist_mete = False
    spec_temp = True

    stacked_list = []
    for folder_BB in folders_BRB:
        spec_files_2018 = []
        env_folder = data_path_input + folder_BB + '/'
        spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*BLU_TSS.tif')
        spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*GRN_TSS.tif')
        spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*NDV_TSS.tif')
        spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*SW1_TSS.tif')
        if len(spec_files_2018) > 1:
            # TODO: create vrt stack instead of "real" stack
            name = env_folder + str(folder_BB) + 'Apr_Okt_stacked.tif'
            create_stack(spec_files_2018, name, n_bands=64, custom_subsetter=range(104, 168 ))
            stacked_list.append(name)
        else:
            pass
        
    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT(data_path_vrt + 'vrt_global.vrt', stacked_list, options=vrt_options)
   
    data_path_input = "X:/SattGruen/Analyse/GLSEG/Raster/S-1/"
    ##########
    # Sentinel 1
    stacked_list = []
    for folder_BB in folders_BRB:
        spec_files_2018 = []
        env_folder = data_path_input + folder_BB + '/'
        spec_files_2018 = Tif_finder(env_folder)
        print(spec_files_2018)
        if len(spec_files_2018) > 1:
            # TODO: create vrt stack instead of "real" stack
            create_stack(spec_files_2018, env_folder + str(folder_BB) + '_stacked.tif', n_bands=2, custom_subsetter=range(1, 3))
            stacked_list.append(env_folder + str(folder_BB) + '_stacked.tif')
        else:
            pass
    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT(data_path_vrt + 'vrt_global.vrt', stacked_list, options=vrt_options)

    """
    import rasterio

    spec_files_2018 = []
    env_folder = "X:/SattGruen/Analyse/GLSEG/Raster/Paulinenaue/X0068_Y0042/"
    name_output_stack = env_folder + 'S1_S2_stack.tif'
    custom_subsetter = None
    #spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\2018*.tif')
    print(len(spec_files_2018))

    #spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*BLU_TSS.tif')
    #spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*GRN_TSS.tif')
    spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*stacked.tif')
    spec_files_2018 = spec_files_2018 + glob.glob(env_folder + '\\*X0068_Y0042_stacked_S1.tif')

    file_list = spec_files_2018
    print(file_list)
    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=64+48)

    # Read each layer and write it to stack
    id_counter = 1
    with rasterio.open(name_output_stack, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            print(id, layer)

            with rasterio.open(layer) as src1:
                print(src1.meta['count'])
                if src1.meta['count'] > 1:
                    custom_subsetter = range(1, src1.meta['count']+1)
                    for i in custom_subsetter:
                        print(i)
                        dst.write_band(id_counter, src1.read(i))
                        id_counter += 1
                        print('BAND=', id_counter)
                else:
                    custom_subsetter = range(1, src1.meta['count']+1)
                    for i in custom_subsetter:
                        print(i)
                        dst.write_band(id_counter, src1.read(i))
                        id_counter += 1
                        print('BAND=', id_counter)
                    custom_subsetter = None
            src0 = None
            src1 = None










