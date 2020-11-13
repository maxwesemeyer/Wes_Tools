import osr
import gdal
import ogr
import os
import re
import numpy as np
import rasterio

from affine import Affine
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
import re
import rasterio.mask
import matplotlib.pyplot as plt
from .Plots_OBIA import *
import cv2
from skimage import exposure
from hubdc.core import *


def Shape_finder(input_path, pattern=".*[s][h][p]{1,2}$"):
    # Alternative...
    #print(glob.glob(data_path + '*.shp'))
    data_path_input = input_path
    file_path_raster = []
    for root, dirs, files in os.walk(data_path_input, topdown=True):
        for file in files:
            if re.match(pattern, file):
                file_path_raster.append(str(root + '/' + file))
            else:
                continue
    return file_path_raster


def Tif_finder(input_path, custom_search_string=".*[t][i][f]{1,2}$"):
    data_path_input = input_path
    file_path_raster = []
    for root, dirs, files in os.walk(data_path_input, topdown=True):
        for file in files:
            if re.match(custom_search_string, file):
                file_path_raster.append(str(root + '/' + file))
            else:
                continue
    return file_path_raster


def WriteArrayToDisk(array, data_path_name_str, gt, polygonite=False, fieldo=None, EPSG=3035):
    #################################
    # write raster file
    # 0 to nan
    # should be 2d
    # img_nan[img_nan == 0] = 255

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    string_tif = data_path_name_str + ".tif"
    # prj =  PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","3035"]]"
    gdal.AllRegister()

    rows = array.shape[0]
    cols = array.shape[1]
    driver = gdal.GetDriverByName('GTiff')
    mean = driver.Create(string_tif, cols, rows, 1, gdal.GDT_Int16)
    mean.SetGeoTransform(gt)
    mean.SetProjection(srs.ExportToWkt())

    band = mean.GetRasterBand(1)

    band.WriteArray(array)
    gdal.SieveFilter(band, None, band, threshold=16)
    if polygonite:
        print('polygonize:....')
        outShapefile = data_path_name_str + "polygonized"
        driver = ogr.GetDriverByName("ESRI Shapefile")

        if os.path.exists(outShapefile + ".shp"):
            driver.DeleteDataSource(outShapefile + ".shp")
        outDatasource = driver.CreateDataSource(outShapefile + ".shp")
        outLayer = outDatasource.CreateLayer(outShapefile, srs=None)
        newField = ogr.FieldDefn('Cluster_nb', ogr.OFTInteger)
        field_2 = ogr.FieldDefn('field_nb', ogr.OFTInteger)
        outLayer.CreateField(newField)
        outLayer.CreateField(field_2)
        band.SetNoDataValue(0)
        band = mean.GetRasterBand(1)
        #band = mean
        gdal.Polygonize(band, None, outLayer, 0, [], callback=None)

        for i in range(outLayer.GetFeatureCount()):
            # print(i)
            feature = outLayer.GetFeature(i)
            feature.SetField('field_nb', fieldo)
            outLayer.CreateFeature(feature)
            feature = None
        outLayer = None
        outDatasource = None
    band = None
    mean = None
    sourceRaster = None


def create_mask_from_ndim(array):
    """

    :param array: should be of shape bands, x, x
    :return:
    """
    out_image_mask = array
    mask = np.any(out_image_mask != 0, axis=0)
    return mask


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def select_bands_sd(array_of_shape, max_valid_pixels_=500, max_bands=500):
    """
    :param array_of_shape: bands, y, x
    :param max_valid_pixels_:
    :param max_bands:
    :return:
    """
    shape_ = array_of_shape.shape
    arg_50 = (-np.nanstd(array_of_shape, axis=(1, 2))).argsort()[:max_bands]
    collected_bands = []
    for args in arg_50:
        valid_pixel = (sum(np.reshape(array_of_shape[args, :, :], (shape_[1] * shape_[2])) > 0))
        if valid_pixel < max_valid_pixels_:
            print('only:', valid_pixel, 'of:', max_valid_pixels_)
        elif len(collected_bands) == max_bands:
            break
        else:
            collected_bands.append(int(args))
    return collected_bands


def image_2_2d(image_of_shape):
    """

    :param image_of_shape: bands, x, y
    :return: x*y, bands
    """
    x, y, z = image_of_shape.shape
    image_2d = np.reshape(image_of_shape, (x, y * z))
    image_2d = np.moveaxis(image_2d.copy(), 0, 1)
    return image_2d




import datetime


def Open_raster_add_meta(inputFile):
    ds = openRasterDataset(inputFile)

    dateStrs = []
    decDates =[]

    for band in ds.bands():
        dateStr = band.metadataItem(key='Date', domain='FORCE', dtype=str)[0:10]
        year = dateStr[0:4]
        month = dateStr[5:7]
        day = dateStr[8:10]
        date = datetime.datetime(np.int(year), np.int(month), np.int(day))
        decDate = (date.timetuple().tm_yday - 1) / 365
        dateStrs.append(dateStr)
        decDates.append(str(np.round(decDate + 2018, 3)))

    ds = gdal.Open(inputFile)
    ds.SetMetadataItem('names', '{EVI}', 'TIMESERIES')
    ds.SetMetadataItem('dates', '{'+ str(', '.join(dateStrs)) + '}', 'TIMESERIES')
    ds.SetMetadataItem('wavelength', '{'+ str(', '.join(decDates)) + '}', 'TIMESERIES')
    ds = None



def Open_raster_add_meta_new_data(inputFile, list_of_dates):
    #ds = openRasterDataset(inputFile)

    dateStrs = []
    decDates = []

    for date in list_of_dates:
        dec_date = toYearFraction(date)
        print(dec_date)
        decDates.append(str(np.round(dec_date, 3)))
        dateStrs.append(date.strftime('%Y-%m-%d'))

    ds = gdal.Open(inputFile)
    ds.SetMetadataItem('names', '{EVI}', 'TIMESERIES')
    ds.SetMetadataItem('dates', '{'+ str(', '.join(dateStrs)) + '}', 'TIMESERIES')
    ds.SetMetadataItem('wavelength', '{'+ str(', '.join(decDates)) + '}', 'TIMESERIES')
    ds = None


def toYearFraction(date):
    from datetime import datetime as dt
    import time
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction
