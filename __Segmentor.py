import sys

import rasterio.mask
from affine import Affine
import matplotlib.pyplot as plt
import os
from shapely import geometry

from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from scipy.ndimage.filters import generic_filter
from scipy import ndimage
import numpy as np
from scipy.stats import mode

# $ conda install --name <conda_env_name> -c <channel_name> <package_name>
sys.path.append("O:/Student_Data/Wesemeyer/Master/conda/myenv/Lib/site-packages/bayseg-master/bayseg")
import bayseg


def filter_function(invalues):
    invalues_mode = mode(invalues, axis=None, nan_policy='omit')
    return invalues_mode[0]


function = lambda array: generic_filter(array, function=filter_function, size=3)


def get_4_tiles_images(image):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides
    width = _ncols / 2
    height = _ncols / 2
    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )


def clahe_nchannels(array):
    """

    :param array: array dim should be x,y,nchannel
    :return: array with shape x, y, nchannel; CLAHE transformed
    """
    # array dim should be x,y,nchannel
    new_array = np.empty(array.shape)
    for i in range(array.shape[2]):
        array_squeezed = array[:, :, i].squeeze()
        new_array[:, :, i] = exposure.equalize_adapthist(array_squeezed, clip_limit=0.001, kernel_size=500)
        # print('clahe', i+1, '/', array.shape[2])
    return new_array


def get_extent(raster):
    """

    :param raster: should be opened in GDAL
    :return: Returns shapely Polygon
    """
    # Get raster geometry
    transform = raster.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft + cols * pixelWidth
    yBottom = yTop + rows * pixelHeight
    rasterGeometry = geometry.box(xLeft, yBottom, xRight, yTop)
    print(xLeft, yTop, transform, cols, rows)
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xLeft, yTop)
    ring.AddPoint(xLeft, yBottom)
    ring.AddPoint(xRight, yTop)
    ring.AddPoint(xRight, yBottom)
    ring.AddPoint(xLeft, yTop)
    rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
    rasterGeometry.AddGeometry(ring)
    """
    return rasterGeometry


def create_mask_from_ndim(array):
    """

    :param array: should be of shape bands, x, x
    :return:
    """
    out_image_mask = array
    mask = np.any(out_image_mask > 0, axis=0)
    return mask


def scfilter(image, iterations, kernel):
    """
    Sine‐cosine filter.
    kernel can be tuple or single value.
    Returns filtered image.
    """
    for n in range(iterations):
        image = np.arctan2(
            ndimage.filters.uniform_filter(np.sin(image), size=kernel),
            ndimage.filters.uniform_filter(np.cos(image), size=kernel))
    return image


def segment_2(string_to_raster, vector_mask, data_path_output=None, beta_coef=1, beta_jump=0.1, n_band=50, into_pca=50,
              custom_subsetter=range(0,80),  MMU=0.05):
    """
    :param string_to_raster: path to raster file
    :param vector_mask: list of fiona geometries
    :param beta_coef: Bayseg parameter; controls autocorrelation of segments
    :param beta_jump: Bayseg parameter
    :param n_band: How many PCs will be used
    :param into_pca: How many bands schould be used for the PCA; By default all
    :param custom_subsetter: In case not to use all the input bands
    :param data_path_output: Where to save the results?
    :param MMU: Minumum Mapping Unit in km²; below that input will be set as one segment
    :return: saves segmented image to disk using the Bayseg
     # the default should be string_to_raster = tss
    # raster_tsi = string_to_raster.replace('TSS', 'TSI')
    range(0, 35 * 14) for coreg stack
    """

    if os.path.exists(data_path_output + 'output'):
        print('output directory already exists')
        #os.rmdir(data_path_output + 'output')
    else:
        os.mkdir(data_path_output + 'output')
    data_patho = data_path_output + 'output'
    #data_patho = 'O:/Student_Data/Wesemeyer/Master/results_new'
    field_counter = "{}{}{}{}{}{}".format(str(into_pca), "_", str(beta_jump), "_", str(n_band), str(np.random.randint(0, 100000)))
    # field_counter = 'stack'
    print(field_counter)
    subsetter_tss = custom_subsetter
    subsetter_tsi = custom_subsetter

    with rasterio.open(string_to_raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, vector_mask, crop=True)
        # with rasterio.open(data_path + 'mask_BB_3035_clip.tif') as src:
        # mask, out_transform_mask = rasterio.mask.mask(src, [shp], crop=True)
    with rasterio.open(string_to_raster) as src:
        out_image_agg, out_mask_agg = rasterio.mask.mask(src, vector_mask, crop=True)
        create_mask = True
        if create_mask:
            mask = create_mask_from_ndim(out_image_agg)

        gt_gdal = Affine.to_gdal(out_transform)
        #################################

        out_image = out_image[subsetter_tss, :, :]
        shape_out = out_image.shape
        max_valid_pixel = np.sum(mask)
        print('Parcel Area:', max_valid_pixel * 100 / 1000000, ' km²')
        if max_valid_pixel * 100 / 1000000 < MMU:
            print('grassland area too small')
            return None
        else:
            w = np.where(out_image < 0)

            out_sub = mask[:, :]
            mask_local = np.where(out_sub <= 0)
            out_image[w] = 0
            out_image_nan = out_image.copy().astype(dtype=np.float)
            out_image_nan[w] = np.nan
            three_band_img = out_image_nan
            three_band_img = np.moveaxis(three_band_img, 0, 2)
            img1 = three_band_img
            print(img1.shape)
            re = np.reshape(img1, (img1.shape[0] * img1.shape[1], img1.shape[2]))
            # scaled_ = RobustScaler(quantile_range=(0.1, 0.9)).fit_transform(re)
            scaled = (MinMaxScaler(feature_range=(0, 10000)).fit_transform(re))
            scaled_shaped = np.reshape(scaled, img1.shape)
            # scaled_shaped = np.square(img1 + 10)
            wh_nan = np.where(np.isnan(scaled_shaped))
            scaled_shaped[wh_nan] = 0

            # slic # args.compactness
            # argmax bands as input for superpixel segmentation
            # -np.nanstd
            arg = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[:3]
            arg_50 = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[0:490]
            arg_10 = []
            for args in arg_50:
                valid_pixel = (sum(np.reshape(out_image[args, :, :], (shape_out[1] * shape_out[2])) > 0))
                if valid_pixel < max_valid_pixel:
                    print('only:', valid_pixel, 'of:', max_valid_pixel)
                elif len(arg_10) == into_pca:
                    break
                else:
                    arg_10.append(int(args))

            print(arg_10)
            ############################################################
            # use tsi instead of tss if too few observations
            #
            if len(arg_10) <= 4:
                field_counter = raster_tsi[41:-4]
                print('TOO FEW OBSERVATIONS USING TSI', field_counter)
                out_image = out_image_agg.copy()
                out_image = out_image[subsetter_tsi, :, :]
                shape_out = out_image.shape
                max_valid_pixel = (sum(np.reshape(mask[:, :], (shape_out[1] * shape_out[2])) > 0))
                print('Parcel Area:', max_valid_pixel * 100 / 1000000, ' km²')
                if max_valid_pixel * 100 / 1000000 < 0.05:
                    print('grassland area too small')
                    return None
                else:
                    w = np.where(out_image < 0)

                    out_sub = mask[:, :]
                    mask_local = np.where(out_sub <= 0)
                    out_image[w] = 0
                    out_image_nan = out_image.copy().astype(dtype=np.float)
                    out_image_nan[w] = np.nan
                    three_band_img = out_image_nan
                    three_band_img = np.moveaxis(three_band_img, 0, 2)
                    img1 = three_band_img
                    print(img1.shape)
                    re = np.reshape(img1, (img1.shape[0] * img1.shape[1], img1.shape[2]))
                    # scaled_ = RobustScaler(quantile_range=(0.1, 0.9)).fit_transform(re)
                    scaled = (MinMaxScaler(feature_range=(0, 10000)).fit_transform(re))
                    scaled_shaped = np.reshape(scaled, img1.shape)
                    # scaled_shaped = np.square(img1 + 10)
                    wh_nan = np.where(np.isnan(scaled_shaped))
                    scaled_shaped[wh_nan] = 0

                    # slic # args.compactness
                    # argmax bands as input for superpixel segmentation
                    # -np.nanstd
                    arg = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[:3]
                    arg_50 = (-np.nanstd(out_image_nan, axis=(1, 2))).argsort()[:800]
                    arg_10 = []
                    for args in arg_50:
                        valid_pixel = (sum(np.reshape(out_image[args, :, :], (shape_out[1] * shape_out[2])) > 0))
                        if valid_pixel < max_valid_pixel:
                            print('only:', valid_pixel, 'of:', max_valid_pixel)
                        elif len(arg_10) == into_pca:
                            break
                        else:
                            arg_10.append(int(args))

                    print(arg_10)
                    # im = clahe_nchannels(scaled_shaped[:, :, arg_10].astype(dtype=np.uint16))
                    im = clahe_nchannels(scaled_shaped[:, :, arg_10].astype(dtype=np.uint16))
            else:
                im = scaled_shaped[:, :, arg_10]

            ############################################################
            # clahe
            # im = clahe_nchannels(scaled_shaped[:, :, arg_10].astype(dtype=np.uint16))
            # im = cv2.medianBlur(im, 5)

            im[im == 0] = np.nan

            scaled_arg_2d = np.reshape(im, (im.shape[0] * im.shape[1], len(arg_10)))

            scaled_arg_2d[np.isnan(scaled_arg_2d)] = 0
            print(scaled_arg_2d)
            #################
            # PCA
            n_comps = n_band
            pca = decomposition.PCA(n_components=n_comps)
            im_pca = pca.fit_transform(scaled_arg_2d)
            print(pca.explained_variance_ratio_)
            print(im_pca.shape)
            #im_pca = im_pca.copy()[:, n_comps]
            image_pca = np.reshape(im_pca, (im.shape[0], im.shape[1], n_comps))
            im_pca[im_pca == 0] = np.nan
            print('IMAGE PCA', image_pca.shape)
            plt.imshow(image_pca)
            plt.show()
            # plt.hist(scaled_arg_2d[:, 0], bins=50)
            # plt.show()

            im[np.isnan(im)] = 0
            # plt.imshow(im[:, :, [0, 1, 2]])
            # plt.show()
            # labels = GaussianMixture(n_components=5).fit_predict(scaled_arg_2d)

            ############################################################
            if max_valid_pixel >= 30000 & max_valid_pixel < 70000:
                n_class = 10
            elif max_valid_pixel <= 10000:
                n_class = 10
            elif max_valid_pixel > 10000 & max_valid_pixel < 30000:
                n_class = 10
            else:
                n_class = 10

            # old 4, 3, 4, 6 for MA now 10
            mino = bayseg.bic(im_pca, n_class)
            # mino = 4
            itero = 500
            # print(mino)
            clf = bayseg.BaySeg(image_pca, mino, beta_init=beta_coef)
            clf.fit(itero, beta_jump_length=beta_jump)

            # shape: n_iter, flat image, n_classes
            # print('PROBSHAPE: ', prob.shape)
            file_str = "{}{}{}".format(data_patho + "/diagnostics", "_stack_", str(field_counter))
            print(file_str)
            ie = clf.diagnostics_plot(transpose=True, save=True, path_to_save=file_str + '.png', ie_return=True)

            labels = clf.labels[-1, :]

            """
            images_iters = []

            for iters in range(itero):
                lo = clf.labels[iters, :]
                lo_img = np.reshape(lo, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
                images_iters.append(lo_img)
                # plt.imshow(lo_img)
                # plt.show()
            """
            # imageio.mimsave(data_path + 'bayseg.gif', images_iters)
            file_str = "{}{}{}".format(data_patho + "/out_labels_pca", str(field_counter), "_")
            file_str_maj = "{}{}{}".format(data_patho + "/out_labels_majority", str(field_counter), "_")
            file_str_ie = "{}{}{}".format(data_patho + "/out_labels_ie_pca", str(field_counter), "_")
            # to save as integer
            labels_img = np.reshape(labels, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
            ie_img = np.reshape(ie, (scaled_shaped.shape[0], scaled_shaped.shape[1])) * 10000
            # prob_img = np.reshape(prob[-1, :, 3], labels_img.shape)

            # labels__img = np.reshape(labels_, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
            labels_img += 1
            labels_img[mask_local] = 0
            labels = labels_img.reshape(im.shape[0] * im.shape[1])
            print(im.shape, labels.shape)
            # labels_img = function(labels_img)
            # plt.imshow(labels_img)
            # plt.show()
            WriteArrayToDisk(labels_img, file_str, gt_gdal, polygonite=True, fieldo=field_counter)
            #
            # WriteArrayToDisk(labels_img, file_str_maj, gt_gdal, polygonite=True, fieldo=field_counter)
            WriteArrayToDisk(ie_img, file_str_ie, gt_gdal, polygonite=False, fieldo=field_counter)


