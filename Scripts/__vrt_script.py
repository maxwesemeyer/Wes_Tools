import glob
import gdal
import os
import re
import sys
sys.path.append("X:/temp/temp_Max/")
import  numpy as np
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.create_vrt import *
import datetime

def main():
    data_path_input = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2017/'
    file_path_raster = Tif_finder(data_path_input, ".*[u][m].*[t][i][f]{1,2}$")
    print(file_path_raster)
    vrt_options = gdal.BuildVRTOptions(separate=False)
    #gdal.BuildVRT(r'\\141.20.140.91/NAS_Rodinia/Croptype/Mowing_2017/vrt/' + 'vrt_SUM.vrt', file_path_raster, options=vrt_options)


def Maj_filter_output():
    data_path_input = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2017/'
    file_path_raster = Tif_finder(data_path_input, ".*[u][m].*[t][i][f]{1,2}$")

    majority_f()
    print(file_path_raster)
    vrt_options = gdal.BuildVRTOptions(separate=False)
    #gdal.BuildVRT(r'\\141.20.140.91/NAS_Rodinia/Croptype/Mowing_2017/vrt/' + 'vrt_SUM.vrt', file_path_raster, options=vrt_options)

def main_2():
    data_path_input = "X:/SattGruen/Analyse/GLSEG/Raster/spectemps1/"
    # data_path_input = r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\AN3_BN1\X0071_Y0039/'
    file_path_raster = Tif_finder(data_path_input, ".*[t][i][f]{1,2}$")  # ".*[B][M].*[t][i][f]{1,2}$"
    print(file_path_raster)
    create_stack(file_path_raster, data_path_input + 'stacked.tif', n_bands=1, custom_subsetter=range(1, 2))
    print('finished')
    s1 = gdal.Open('X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/X0071_Y0039/stacked.tif')
    arr_mean = s1.ReadAsArray().mean(axis=0)
    arr_sd = s1.ReadAsArray().std(axis=0)

    arr_q_25 = np.quantile(s1.ReadAsArray(), 0.05, axis=0)
    arr_Q_75 = np.quantile(s1.ReadAsArray(), 0.975, axis=0)

    WriteArrayToDisk(arr_mean.squeeze() * 100, 'X:/SattGruen/Analyse/GLSEG/Raster/Ramin_S1/stacked_mean.tif',
                     s1.GetGeoTransform())
    WriteArrayToDisk(arr_sd.squeeze() * 100, 'X:/SattGruen/Analyse/GLSEG/Raster/Ramin_S1/stacked_sd.tif',
                     s1.GetGeoTransform())

    WriteArrayToDisk(arr_q_25.squeeze() * 100, 'X:/SattGruen/Analyse/GLSEG/Raster/Ramin_S1/arr_q_25.tif',
                     s1.GetGeoTransform())
    WriteArrayToDisk(arr_Q_75.squeeze() * 100, 'X:/SattGruen/Analyse/GLSEG/Raster/Ramin_S1/arr_Q_75.tif',
                     s1.GetGeoTransform())


def main_stack(folders_BRB):

    for folder in folders_BRB:
        force_tile = folder.split('/')[-1]
        print(force_tile)
        data_path_input = folder + '/'
        file_path_raster = Tif_finder(data_path_input, ".*[t][i][f]{1,2}$")  # ".*[B][M].*[t][i][f]{1,2}$"
        if not file_path_raster:
            continue
        else:
            out_string = data_path_input + 'S1_stacked_' + force_tile + '.tif'
            create_stack(file_path_raster,out_string , n_bands=1, custom_subsetter=range(1, 2))


def coherence_stack(in_path, out_path):
    Tifs = Tif_finder(in_path)
    force_tiles = []
    asc_dec_list = []
    # get a list of all force tiles that are in folder
    for raster in Tifs:
        file_name = raster.split('/')[-1]
        file_name_substr = file_name.split('_')
        asc_dec_list.append(file_name_substr[-1].split('.')[0])
        force_tiles.append(file_name_substr[0] + '_' + file_name_substr[1])

    # iterate through force tiles and Asc/Dec and create a list of all rasters that should be written in one stack
    # also create a list of the respective dates for the MetaData
    for force_tile in np.unique(force_tiles):
        subs_force = [k for k in Tifs if force_tile in k]
        for asc_dec in np.unique(asc_dec_list):
            try:
                subs_force_asc = [k for k in subs_force if asc_dec in k]
                date_list = []
                for path_strings in subs_force_asc:
                    print(path_strings)
                    file_name = path_strings.split('/')[-1]
                    file_name_substr = file_name.split('_')
                    dateo = datetime.datetime.strptime(str(file_name_substr[2]), '%Y%m%d')
                    print(dateo)
                    # append every date twice because we have two bands per image... the easy way
                    date_list.append(dateo)
                    #date_list.append(dateo)
                    # print(dateo.strftime('%Y-%m-%d'))
                out_string = out_path + force_tile + '/coherence_stack_2018_vv' + asc_dec + '.tif'
                create_stack(subs_force_asc, out_string, n_bands=1, custom_subsetter=range(2,3))
                Open_raster_add_meta_new_data(out_string, date_list)
                print(date_list, np.argsort(date_list))
                print(subs_force_asc, 'Right subset')
            except:
                continue


def rainstack(path_to_RADOLAN, out_path):

    Radolan_list = Tif_finder(path_to_RADOLAN, custom_search_string=".*[i][l][y].*[e][n][v][i]{1,2}$")
    date_list = []
    for file in Radolan_list:
        date = file.split('_')[1].split('//')[-1]

        date = datetime.datetime.strptime(str(date), '%Y%m%d')
        date_list.append(date)
    #create_stack(Radolan_list, out_path, n_bands=1, custom_subsetter=range(1,2))

    for tile in Tif_finder( r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\RADOLAN\Tiles/'):

        Open_raster_add_meta_new_data(tile, date_list)





if __name__ == '__main__':
    #in_ = r'\\141.20.140.91/SAN/_ProjectsII/Grassland/SattGruen/Analyse/Mowing_detection/Data/Raster/coherenceFORCETiles'
    #out = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\SattGruen\Analyse\Mowing_detection\Data/Raster/AN0_BN0/'
    #coherence_stack(in_, out)
    out = r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\RADOLAN/Radolan_stacked.tif'
    rainstack(r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\RADOLAN/Daily/', out)
    """
    
    folders_BRB = [x[0] for x in os.walk(r'X:\SattGruen\Analyse\Mowing_detection\Data\Raster\AN3_BN1\S-1/')]
    print(folders_BRB[1:])
    main_stack(folders_BRB[1:])
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



"""






