import os
import glob
from osgeo import gdal
from joblib import Parallel, delayed
from osgeo import gdal
import re
import rasterio
from .__utils import *

def create_stack(path_to_overlapping_rasters, name_output_stack, n_bands=70, custom_subsetter=range(1,70)):
    """
    not finished yet...
    :param path_to_overlapping_rasters: list of paths to rasters
    :param n_bands: number of bands per raster; will be multiplied by number of files
    :return:
    """
    file_list = path_to_overlapping_rasters
    print(file_list)
    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(file_list)*n_bands)

    # Read each layer and write it to stack
    id_counter = 1
    with rasterio.open(name_output_stack, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            print(id, layer)

            with rasterio.open(layer) as src1:
                for i in custom_subsetter:
                    print(i)
                    dst.write_band(id_counter, src1.read(i))
                    id_counter += 1
                    print('BAND=', id_counter)
            src0 = None
            src1 = None


def merge_vrt_tiles(folder_stacked_vrt, folder_stacked_vrt_global):

    # create global vrt 2018
    tiles_2018 = []
    tiles_2018 = tiles_2018 + (glob.glob(folder_stacked_vrt + '*.vrt'))
    print(tiles_2018)
    #tiles_2018 = sorted(tiles_2018, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    filename_2018 = folder_stacked_vrt_global + 'specclim' + "_2018.vrt"
    print(filename_2018)
    vrt_options = gdal.BuildVRTOptions(separate=False)
    gdal.BuildVRT(filename_2018, tiles_2018, options=vrt_options)


def mixs(num):
    try:
        ele = int(num)
        return (0, ele, '')
    except ValueError:
        return (1, num, '')

