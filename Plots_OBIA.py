from matplotlib import colors
import numpy as np
import fiona
import math
import rasterio
import matplotlib.pyplot as plt
import dictances
import itertools
import pandas as pd


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def plot_shapefile(vector_data, raster_data, own_segmentation=False):
    # Create a ListedColormap with only the color green specified
    cmap = colors.ListedColormap(['green'])
    # Use the `set_bad` property of `colormaps` to set all the 'bad' data to red
    cmap.set_bad(color='red')

    raster_ = raster_data
    print(vector_data)
    if own_segmentation:
        tif_str = vector_data.replace('polygonized.shp', '.tif')
        r = gdal.Open(tif_str)
        r_arr = r.ReadAsArray()

    shapes = []
    with fiona.open(vector_data) as shapefile:
        if own_segmentation:
            feature_list = []
            for features in shapefile:
                if features["properties"]['Cluster_nb'] != 0 and features["properties"]['field_nb'] != None:
                    shapes.append(features["geometry"])
                    values_view = features["properties"].values()
                    value_iterator = iter(values_view)
                    first_value = next(value_iterator)
                    feature_list.append(first_value)
                else:
                    continue
        else:
            feature_list = []
            for features in shapefile:
                #feature_list.append(features["properties"]['field_nb'])
                values_view = features["properties"].values()
                value_iterator = iter(values_view)
                first_value = next(value_iterator)
                feature_list.append(first_value)

            shapes = [feature["geometry"] for feature in shapefile]

    i = 1
    grid_size = int(math.ceil(np.sqrt(len(shapes))))
    print(grid_size)
    for shp in shapes:

        with rasterio.open(raster_) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shp], crop=True)
            out_image = out_image[range(35),:,:]
            out_image = out_image.astype(dtype=np.float)
            out_image[out_image < 0] = np.nan
            print(out_image.shape)
            row_mean = (np.nanmean(out_image, axis=(1, 2))) / 10000
            row_sd = (np.nanstd(out_image, axis=(1, 2))) / 10000
            # row_mean[np.isnan(row_mean)] = 0

            ##############
            # interpolated
            nans, x = nan_helper(row_mean)
            row_mean[nans] = np.interp(x(nans), x(~nans), row_mean[~nans])

            nans, x = nan_helper(row_sd)
            row_sd[nans] = np.interp(x(nans), x(~nans), row_sd[~nans])

            x = np.linspace(1, len(row_mean), len(row_mean))
            plt.subplot(grid_size, grid_size, i)
            plt.title(str(feature_list[i-1]))
            plt.errorbar(x, row_mean, row_sd)

            # create list of length nans and convert it to array then set colors
            color = np.array([str(1)] * len(nans))
            color[nans] = 'red'
            color[~nans] = 'green'

            #plt.scatter(x, row_mean, s=5, c=color)
            i += 1
    plt.show()
    if own_segmentation:
        plt.imshow(r_arr)
        plt.show()


def DistancoTron(vector_data, raster_data, own_segmentation=False):
    # Create a ListedColormap with only the color green specified
    cmap = colors.ListedColormap(['green'])
    # Use the `set_bad` property of `colormaps` to set all the 'bad' data to red
    cmap.set_bad(color='red')

    raster_ = raster_data
    print(vector_data)
    if own_segmentation:
        tif_str = vector_data.replace('polygonized.shp', '.tif')
        r = gdal.Open(tif_str)
        r_arr = r.ReadAsArray()

    shapes = []
    with fiona.open(vector_data) as shapefile:
        if own_segmentation:
            feature_list = []
            for features in shapefile:
                if features["properties"]['Cluster_nb'] != 0 and features["properties"]['field_nb'] != None:
                    shapes.append(features["geometry"])
                    values_view = features["properties"].values()
                    value_iterator = iter(values_view)
                    first_value = next(value_iterator)
                    feature_list.append(first_value)
                else:
                    continue
        else:
            feature_list = []
            for features in shapefile:
                #feature_list.append(features["properties"]['field_nb'])
                values_view = features["properties"].values()
                value_iterator = iter(values_view)
                first_value = next(value_iterator)
                feature_list.append(first_value)


            shapes = [feature["geometry"] for feature in shapefile]

    i = 1
    grid_size = int(math.ceil(np.sqrt(len(shapes))))
    print(grid_size)
    raster_data_list = []
    for shp in shapes:

        with rasterio.open(raster_) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shp], crop=True)
            out_image = out_image[range(35),:,:]
            out_image = out_image.astype(dtype=np.float)
            out_image[out_image < 0] = np.nan
            print(out_image.shape)
            row_mean = (np.nanmean(out_image, axis=(1, 2))) / 10000
            row_sd = (np.nanstd(out_image, axis=(1, 2))) / 10000
            # row_mean[np.isnan(row_mean)] = 0

            ##############
            # interpolated
            nans, x = nan_helper(row_mean)
            row_mean[nans] = np.interp(x(nans), x(~nans), row_mean[~nans])

            nans, x = nan_helper(row_sd)
            row_sd[nans] = np.interp(x(nans), x(~nans), row_sd[~nans])
            ###############
            # calc distance
            raster_data_list.append(row_mean)
            dict_row_mean = {i: row_mean[i] for i in range(0, len(row_mean))}

            if i % 2 == 0:

                di = dictances.bhattacharyya(dict_row_mean_old, dict_row_mean)
                print(di)
            dict_row_mean_old = dict_row_mean
            """
            x = np.linspace(1, len(row_mean), len(row_mean))
            plt.subplot(grid_size, grid_size, i)
            plt.title(str(feature_list[i-1]))
            plt.errorbar(x, row_mean, row_sd)

            # create list of length nans and convert it to array then set colors
            color = np.array([str(1)] * len(nans))
            color[nans] = 'red'
            color[~nans] = 'green'
            
            #plt.scatter(x, row_mean, s=5, c=color)
            """
            i += 1


def aggregator(raster_NDV, shapei, indexo=np.random.randint(0, 10000), raster_clim=None, ie_raster=None, subsetter=range(146, 218)):
    geo = shapei.geometry
    index = indexo
    print(index)
    if raster_clim:
        with rasterio.open(raster_clim) as src:
            out_image, out_transform = rasterio.mask.mask(src, [geo], crop=True)

            out_image = out_image.astype(dtype=np.float)
            out_image[out_image < 0] = np.nan
            row_mean_clim = (np.nanmean(out_image, axis=(1, 2)))
            row_mean_clim[np.isnan(row_mean_clim)] = 0
    if ie_raster:
        with rasterio.open(ie_raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, [geo], crop=True)

            out_image = out_image.astype(dtype=np.float)
            out_image[out_image < 0] = np.nan
            row_mean_ie = [np.nanmean(out_image)]
            if np.isnan(row_mean_ie):
                row_mean_ie = [0]

    with rasterio.open(raster_NDV) as src:
        out_image, out_transform = rasterio.mask.mask(src, [geo], crop=True)
        print(out_image.shape)
        subsetter_tsi = subsetter

        if subsetter is None:
            out = out_image[:, :, :]
        else:
            out = out_image[subsetter_tsi, :, :]

        out = out.astype(dtype=np.float)
        out[out < 0] = np.nan
        row_mean = (np.nanmean(out, axis=(1, 2))) / 10000
        row_mean[np.isnan(row_mean)] = 0
        print(row_mean)
        # sd
        row_sd = (np.nanstd(out, axis=(1, 2))) / 10000
        row_sd[np.isnan(row_sd)] = 0
        if raster_clim and ie_raster:
            ab = itertools.chain(row_mean, row_mean_clim, row_sd, row_mean_ie)
        else:
            ab = itertools.chain(row_mean, row_sd)
        l = list(ab)
        l.append(index)
        Serio = pd.Series(l)
        print(Serio)
        return Serio