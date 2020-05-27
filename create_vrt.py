import os
import glob
from osgeo import gdal
from joblib import Parallel, delayed
from osgeo import gdal
import re
import rasterio


def create_stack(path_to_overlapping_rasters, n_bands=70):
    """
    not finished yet...
    :param path_to_overlapping_rasters: list of paths to rasters
    :param n_bands: number of bands per raster; will be multiplied by number of files
    :return:
    """
    file_list = path_to_overlapping_rasters

    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(file_list)*n_bands)

    # Read each layer and write it to stack
    id_counter = 1
    with rasterio.open('Z:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/stack.tif', 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            print(id, layer)

            with rasterio.open(layer) as src1:
                for i in range(1,35):
                    print(i)
                    dst.write_band(id_counter, src1.read(i))
                    id_counter += 1
                    print('BAND=', id_counter)


def merge_vrt_tiles(folder_stacked_vrt, folder_stacked_vrt_global):

    # create global vrt 2018
    tiles_2018 = []
    tiles_2018 = tiles_2018 + (glob.glob(folder_stacked_vrt + '*.vrt'))
    print(tiles_2018)
    #tiles_2018 = sorted(tiles_2018, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    filename_2018 = folder_stacked_vrt_global +  'specclim' + "_2018.vrt"
    print(filename_2018)
    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT(filename_2018, tiles_2018, options=vrt_options)


def mixs(num):
    try:
        ele = int(num)
        return (0, ele, '')
    except ValueError:
        return (1, num, '')

folders_BRB = {"X0065_Y0040", "X0065_Y0041", "X0066_Y0040", "X0066_Y0041", "X0066_Y0042", "X0067_Y0040", "X0067_Y0041",
               "X0067_Y0042", "X0067_Y0043", "X0067_Y0044", "X0067_Y0045", "X0068_Y0040", "X0068_Y0041", "X0068_Y0042",
               "X0068_Y0043", "X0068_Y0044", "X0068_Y0045", "X0069_Y0040", "X0069_Y0041", "X0069_Y0042", "X0069_Y0043",
               "X0069_Y0044", "X0069_Y0045", "X0069_Y0046", "X0069_Y0047", "X0070_Y0039", "X0070_Y0040", "X0070_Y0041",
               "X0070_Y0042", "X0070_Y0043", "X0070_Y0044", "X0070_Y0045", "X0070_Y0046", "X0070_Y0047", "X0071_Y0039",
               "X0071_Y0040", "X0071_Y0041", "X0071_Y0042", "X0071_Y0043", "X0071_Y0044", "X0071_Y0045", "X0071_Y0046",
               "X0071_Y0047", "X0072_Y0040", "X0072_Y0042", "X0072_Y0043", "X0072_Y0044", "X0072_Y0045", "X0072_Y0046",
               "X0072_Y0047", "X0073_Y0044", "X0073_Y0045", "X0073_Y0046"}
data_path_input = 'Z:/'
# find all files in the given directory containing either MODIS or bsq
file_path_raster = []
for root, dirs, files in os.walk(data_path_input, topdown=True):
    dirs[:] = [d for d in dirs if d in folders_BRB]
    for file in files:
        if re.match("^2016.*[V][_][T][S][I].[t][i][f]{1,2}$", file):
            file_path_raster.append(str(root + '/' + file))
        else:
            continue

topo = False
soil = False

temp_clim = False
prec_clim = False

temp_mete = False
prec_mete = False
smoist_mete = False


# rad
for folder_BB in folders_BRB:

    env_folder = "Z:/" + folder_BB
    print(env_folder)
    spec_files_2018 = []
    if topo == True:

        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\srtm_10m.tif'))
        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\slope_10m.tif'))
        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\aspect_10m.tif'))

    if soil == True:

        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\CECSOL_M_sl1_10m.tif'))
        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\ORCDRC_M_sl1_10m.tif'))
        spec_files_2018 = spec_files_2018 + (glob.glob(env_folder + '\\PHIKCL_M_sl1_10m.tif'))

    if temp_mete == True:
        #print(sorted(glob.glob(env_folder + '\\TAMM_*2017_01_10m.tif'))[2:11])

        spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\TAMM_*2018_01_10m.tif'), key=mixs)[2:11]
        print(spec_files_2018)

    if prec_mete == True:
        #print(sorted(glob.glob(env_folder + '\\rad_2017*'))[2:11])

        spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\rad_2018*'), key=mixs)[2:11]


    if smoist_mete == True:
        #print(sorted(glob.glob(env_folder + '\\grids_germany_monthly_soil_moist_2017*'))[2:11])

        spec_files_2018 = spec_files_2018 + sorted(glob.glob(env_folder + '\\grids_germany_monthly_soil_moist_2018*'), key=mixs)[2:11]

    vrt_options = gdal.BuildVRTOptions(separate = True)
    vrt_string = "{}{}{}{}{}".format("Z:/BB_vrt/", folder_BB , "/Precip_2018_", str(folder_BB), ".vrt")
    print(len(spec_files_2018))
    print(vrt_string)
    if not os.path.exists("Z:/BB_vrt/" + folder_BB):
        os.mkdir("Z:/BB_vrt/" + folder_BB)
    gdal.BuildVRT(vrt_string, spec_files_2018, options=vrt_options)


clims_raster = []
for root, dirs, files in os.walk('Z:/BB_vrt/'):
    for file in files:
        if re.match("^[P].*", file):
            clims_raster.append(str(root + '/' + file ))
        else:
            continue
print(clims_raster)
#merge_vrt_tiles('Z:/BB_vrt/', 'Z:/BB_vrt_stack/')


vrt_options = gdal.BuildVRTOptions(separate=False)
gdal.BuildVRT('Z:/BB_vrt_global/spec_clim_2018.vrt', clims_raster, options=vrt_options)



vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=False)
gdal.BuildVRT('Z:/BB_vrt/BRB_precipitation.vrt', file_path_precip, options=vrt_options)


data_path_input = 'O:/Student_Data/Wesemeyer/Master/results'
# find all files in the given directory containing either MODIS or bsq


data_path_input = 'Z:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/'
file_path_raster = Tif_finder(data_path_input)





