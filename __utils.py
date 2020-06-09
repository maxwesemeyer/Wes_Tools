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



def Shape_finder(input_path):
    # Alternative...
    #print(glob.glob(data_path + '*.shp'))
    data_path_input = input_path
    file_path_raster = []
    for root, dirs, files in os.walk(data_path_input, topdown=True):
        for file in files:
            if re.match(".*[s][h][p]{1,2}$", file):
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


def prepare_data(raster_l, vector_geom, custom_subsetter=range(5,65), n_band=11, MMU=0.05, PCA=True):
    shp = [vector_geom.geometry]
    subsetter_tsi = custom_subsetter
    try:
        with rasterio.open(raster_l) as src:
            out_image, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=0)
            mask = create_mask_from_ndim(out_image)
            #out_image = out_image + 100000
            out_image = out_image*mask
            gt_gdal = Affine.to_gdal(out_transform)
            #################################
            out_meta = src.meta
            out_image = out_image.copy() / 10000
            out_image = out_image[subsetter_tsi, :, :]
            shape_out = out_image.shape
            max_valid_pixel = (sum(np.reshape(mask[:, :], (shape_out[1] * shape_out[2])) > 0))
            print(max_valid_pixel)
            print('Parcel Area:', max_valid_pixel * 100 / 1000000, ' km²')
            if max_valid_pixel * 100 / 1000000 < MMU:
                print('pass, MMU')
                return None, None, None, None
            else:
                w = np.where(out_image < 0)
                out_sub = mask[:, :]
                mask_local = np.where(out_sub == 0)
                out_image[w] = 0
                out_image_nan = out_image.copy().astype(dtype=np.float)
                out_image_nan[w] = np.nan

                three_band_img = out_image_nan
                img1 = np.moveaxis(three_band_img, 0, 2)

                re = np.reshape(img1, (img1.shape[0] * img1.shape[1], img1.shape[2]))
                # re_scale = RobustScaler(quantile_range=(0.8, 1)).fit_transform(re)
                #scaled = (MinMaxScaler(feature_range=(0, 255)).fit_transform(re))
                scaled = re
                scaled_shaped = np.reshape(scaled, (img1.shape))
                # scaled_shaped = np.square(img1+10)

                ###########
                # selects bands which have only valid pixels
                scaled_shaped[np.where(scaled_shaped==0)] = np.nan
                std_glob = np.nanstd(scaled_shaped, axis=(1, 2))
                print('global:', sum(std_glob))
                arg_10 = select_bands_sd(np.moveaxis(scaled_shaped, 2, 0), max_valid_pixels_=max_valid_pixel)
                wh_nan = np.where(np.isnan(scaled_shaped))
                scaled_shaped[wh_nan] = 0

                im = scaled_shaped[:, :, arg_10]
                im[im == 0] = np.nan
                scaled_arg_2d = np.reshape(im, (im.shape[0] * im.shape[1], len(arg_10)))
                im[np.isnan(im)] = 0
                scaled_arg_2d[np.isnan(scaled_arg_2d)] = 0

                if PCA:
                    print(arg_10)
                    #################
                    # PCA
                    import matplotlib.pyplot as plt

                    if len(arg_10) > n_band:
                        n_comps = n_band
                    else:
                        n_comps = len(arg_10)
                    pca = decomposition.PCA(n_components=n_comps)
                    im_pca_2d = pca.fit_transform(scaled_arg_2d)
                    print(pca.explained_variance_ratio_)
                    image_pca = np.reshape(im_pca_2d, (im.shape[0], im.shape[1], n_comps))
                    im_pca_2d[im_pca_2d == 0] = np.nan
                    print('IMAGE PCA', image_pca.shape)
                    #plt.imshow(image_pca[:,:,:3])
                    #plt.show()
                    return image_pca, im_pca_2d, mask_local, gt_gdal
                else:
                    print(arg_10)
                    if im[:, :, :].shape[2] < n_band:
                        print('n_band parameter bigger than bands available; used all available bands')
                        return im, scaled_arg_2d, mask_local, gt_gdal
                    else:
                        print('no pca, used: ', n_band, ' bands')
                        return im[:, :, :n_band], scaled_arg_2d[:, :n_band], mask_local, gt_gdal
    except:
        print('Maybe input shapes did not overlap; Also check subsetter')
        return None, None, None, None

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
