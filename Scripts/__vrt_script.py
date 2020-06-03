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

    folders_BRB = {"X0065_Y0040", "X0065_Y0041", "X0066_Y0040", "X0066_Y0041", "X0066_Y0042", "X0067_Y0040", "X0067_Y0041",
                   "X0067_Y0042", "X0067_Y0043", "X0067_Y0044", "X0067_Y0045", "X0068_Y0040", "X0068_Y0041", "X0068_Y0042",
                   "X0068_Y0043", "X0068_Y0044", "X0068_Y0045", "X0069_Y0040", "X0069_Y0041", "X0069_Y0042", "X0069_Y0043",
                   "X0069_Y0044", "X0069_Y0045", "X0069_Y0046", "X0069_Y0047", "X0070_Y0039", "X0070_Y0040", "X0070_Y0041",
                   "X0070_Y0042", "X0070_Y0043", "X0070_Y0044", "X0070_Y0045", "X0070_Y0046", "X0070_Y0047", "X0071_Y0039",
                   "X0071_Y0040", "X0071_Y0041", "X0071_Y0042", "X0071_Y0043", "X0071_Y0044", "X0071_Y0045", "X0071_Y0046",
                   "X0071_Y0047", "X0072_Y0040", "X0072_Y0042", "X0072_Y0043", "X0072_Y0044", "X0072_Y0045", "X0072_Y0046",
                   "X0072_Y0047", "X0073_Y0044", "X0073_Y0045", "X0073_Y0046"}

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
            create_stack(spec_files_2018, env_folder + str(folder_BB) + '_stacked.tif', n_bands=75, custom_subsetter=range(90, 165))
            stacked_list.append(env_folder + str(folder_BB) + '_stacked.tif')
        else:
            pass


    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT(data_path_vrt + 'vrt_global.vrt', stacked_list, options=vrt_options)

    """
    data_path = "X:/SattGruen/Analyse/GLSEG/Raster/landsat_sentinel/X0068_Y0042/"
    list_raster = Tif_finder(data_path, "^2016.*[S][.][t][i][f]{1,2}$")
    print(list_raster)
        create_stack(list_raster, data_path + 'stacked.tif', n_bands=75 ,custom_subsetter=range(90,165))
    """
"""
    for folder_BB in folders_BRB:

        env_folder = data_path_input + folder_BB + '/'

        spec_files_2018 = []

        if spec_temp:
            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\*BLU_TSS.tif'), key=mixs)
            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\*GRN_TSS.tif'), key=mixs)
            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\*NDV_TSS.tif'), key=mixs)
            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\*SW1_TSS.tif'), key=mixs)

            print(spec_files_2018)

        if topo:

            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\srtm_10m.tif'))
            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\slope_10m.tif'))
            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\aspect_10m.tif'))

        if soil:

            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\CECSOL_M_sl1_10m.tif'))
            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\ORCDRC_M_sl1_10m.tif'))
            spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\PHIKCL_M_sl1_10m.tif'))

        if temp_mete:
            #print(sorted(glob.glob(env_folder + '\\TAMM_*2017_01_10m.tif'))[2:11])

            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\TAMM_*2018_01_10m.tif'), key=mixs)[2:11]
            print(spec_files_2018)

        if prec_mete:
            #print(sorted(glob.glob(env_folder + '\\rad_2017*'))[2:11])

            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\rad_2018*'), key=mixs)[2:11]


        if smoist_mete:
            #print(sorted(glob.glob(env_folder + '\\grids_germany_monthly_soil_moist_2017*'))[2:11])

            spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\grids_germany_monthly_soil_moist_2018*'), key=mixs)[2:11]

        vrt_options = gdal.BuildVRTOptions(separate = True)
        vrt_string = "{}{}{}{}{}".format(data_path_vrt, folder_BB , "/Precip_2018_", str(folder_BB), ".vrt")
        print(len(spec_files_2018))
        print(vrt_string)
        if not os.path.exists(data_path_vrt + folder_BB):
            os.mkdir(data_path_vrt + folder_BB)
        gdal.BuildVRT(vrt_string, spec_files_2018, options=vrt_options)


    clims_raster = []
    for root, dirs, files in os.walk(data_path_vrt):
        for file in files:
            if re.match("^[P].*", file):
                clims_raster.append(str(root + '/' + file ))
            else:
                continue
    print(clims_raster)
    #merge_vrt_tiles(data_path_vrt, 'Z:/BB_vrt_stack/')


    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT('Z:/BB_vrt_global/spec_clim_2018.vrt', clims_raster, options=vrt_options)



    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=False)
    gdal.BuildVRT('Z:/BB_vrt/BRB_precipitation.vrt', file_path_precip, options=vrt_options)
"""







