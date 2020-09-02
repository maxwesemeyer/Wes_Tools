import sys
from shapely import geometry
from skimage import exposure
from scipy.ndimage.filters import generic_filter
from scipy import ndimage
from scipy.stats import mode
from .__utils import *
import bayseg


class segmentation_BaySeg:

    def __init__(self, beta_coef=1, beta_jump=0.1, n_band=50, custom_subsetter=range(0, 80), MMU=0.05, PCA=True,
                 into_pca='all', n_class=4, iterations=70, _filter='bilateral', neighbourhood="4p"):
        self.beta_coef = beta_coef
        self.beta_jump = beta_jump
        self.MMU = MMU
        self.subsetter = custom_subsetter
        self.n_band = n_band
        self.PCA = PCA
        self.into_pca = into_pca
        self.n_class = n_class
        self.iterations = iterations
        self.stncl = neighbourhood
        self._filter = _filter

    def get_edges(self, three_d_image):
        from skimage import feature, filters
        from skimage.segmentation import join_segmentations
        from skimage.measure import label
        preceding_segmentation = None
        for bands in range(1):
            edges = filters.roberts(three_d_image[:, :, bands].squeeze()).astype(int)

            if preceding_segmentation is not None:
                preceding_segmentation = join_segmentations(edges, preceding_segmentation)
            else:
                preceding_segmentation = edges

        print('Percentiles', np.percentile(preceding_segmentation, q=50))
        print('Percentiles', np.percentile(preceding_segmentation, q=90))
        preceding_segmentation[preceding_segmentation > 300] = 0
        preceding_segmentation[preceding_segmentation < 22] = 0

        preceding_segmentation = majority_f(preceding_segmentation)
        contours = label(preceding_segmentation)
        np.histogram(preceding_segmentation)
        print(np.unique(preceding_segmentation))
        plt.imshow(preceding_segmentation)
        plt.show()
        plt.imshow(contours)
        plt.show()

        three_d_image = np.stack((three_d_image[:, :, 0].squeeze(), contours), axis=2)
        return three_d_image

    def prepare_data(self, raster_l, vector_geom):
        shp = [vector_geom.geometry]
        """
        with fiona.open(vector_geom, "r") as shapefile:
            shp = [feature["geometry"] for feature in shapefile]
        """
        subsetter_tsi = self.subsetter
        try:
            with rasterio.open(raster_l) as src:
                out_image, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=0)
                mask = create_mask_from_ndim(out_image)
                out_image[out_image < 0] = abs(out_image[out_image < 0])
                out_image = out_image * mask
                gt_gdal = Affine.to_gdal(out_transform)
                #################################
                out_meta = src.meta
                out_image = out_image.copy() / 10000
                out_image = out_image[subsetter_tsi, :, :]
                shape_out = out_image.shape
                max_valid_pixel = (sum(np.reshape(mask[:, :], (shape_out[1] * shape_out[2])) > 0))
                print('Parcel Area:', max_valid_pixel * 100 / 1000000, ' km²')

                import scipy
                import matplotlib.pyplot as plt
                mass_center = scipy.ndimage.measurements.center_of_mass(mask)
                print(mass_center)
                print(mask[int(mass_center[0]), int(mass_center[1])])


                if max_valid_pixel * 100 / 1000000 < self.MMU:
                    print('pass, MMU')
                    MMU_fail = True
                elif not mask[int(mass_center[0]), int(mass_center[1])]:
                    print('pass, shape fail')
                    MMU_fail = True
                else:
                    MMU_fail = False
                w = np.where(out_image < 0)
                out_sub = mask[:, :]
                mask_local = np.where(out_sub == 0)
                out_image[w] = 0
                out_image_nan = out_image.copy().astype(dtype=np.float)
                out_image_nan[w] = np.nan

                if MMU_fail:
                    return np.moveaxis(out_image, 0, 2), None, mask_local, gt_gdal, MMU_fail
                del out_image
                three_band_img = out_image_nan
                del out_image_nan
                img1 = np.moveaxis(three_band_img, 0, 2)

                re = np.reshape(img1, (img1.shape[0] * img1.shape[1], img1.shape[2]))
                # re_scale = RobustScaler(quantile_range=(0.8, 1)).fit_transform(re)
                if self._filter == 'clahe':
                    scaled = (MinMaxScaler(feature_range=(0, 1)).fit_transform(re))
                else:
                    scaled = (MinMaxScaler(feature_range=(0, 255)).fit_transform(re))
                # scaled = re
                scaled_shaped = np.reshape(scaled, (img1.shape))
                # scaled_shaped = np.square(img1+10)

                ###########
                # selects bands which have only valid pixels
                scaled_shaped[np.where(scaled_shaped == 0)] = np.nan

                arg_10 = select_bands_sd(np.moveaxis(scaled_shaped, 2, 0), max_valid_pixels_=max_valid_pixel)

                # change back ... needs to be always the same when model should be reuseable
                #arg_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ##################### !!
                wh_nan = np.where(np.isnan(scaled_shaped))
                scaled_shaped[wh_nan] = 0
                im = scaled_shaped[:, :, arg_10]
                im[im == 0] = np.nan
                scaled_arg_2d = np.reshape(im, (im.shape[0] * im.shape[1], len(arg_10)))
                im[np.isnan(im)] = 0
                scaled_arg_2d[np.isnan(scaled_arg_2d)] = 0
                if self._filter == 'bilateral':
                    im = np.float32(im)
                    im = bilateral_nchannels(im)
                elif self._filter == 'clahe':
                    im = clahe_nchannels(im)
                else:
                    None
                if self.PCA:
                    print(arg_10)
                    #################
                    # PCA
                    import matplotlib.pyplot as plt

                    if len(arg_10) > self.n_band:
                        n_comps = self.n_band
                        if self.into_pca == 'all':
                            None
                        else:
                            scaled_arg_2d = scaled_arg_2d[:, :self.into_pca]
                    else:
                        n_comps = len(arg_10)
                    pca = decomposition.PCA(n_components=n_comps)
                    im_pca_2d = pca.fit_transform(scaled_arg_2d)
                    print(pca.explained_variance_ratio_)
                    image_pca = np.reshape(im_pca_2d, (im.shape[0], im.shape[1], n_comps))
                    im_pca_2d[im_pca_2d == 0] = np.nan
                    print('IMAGE PCA', image_pca.shape)
                    # plt.imshow(image_pca[:,:,:3])
                    # plt.show()
                    return image_pca, im_pca_2d, mask_local, gt_gdal, MMU_fail
                else:
                    print(arg_10)
                    if im[:, :, :].shape[2] < self.n_band:
                        print('n_band parameter bigger than bands available; used all available bands')
                        return im, scaled_arg_2d, mask_local, gt_gdal, MMU_fail
                    else:
                        print('no pca, used: ', self.n_band, ' bands')
                        return im[:, :, :self.n_band], scaled_arg_2d[:, :self.n_band], mask_local, gt_gdal, MMU_fail

        except:
            print('Maybe input shapes did not overlap; Also check subsetter; Is N_band paramter a list?', self.n_band)
            return None, None, None, None, None

    def segment_2(self, string_to_raster, vector_geom, data_path_output, indexo=np.random.randint(0, 100000)):
        """
        :param string_to_raster: path to raster file
        :param vector_geom: list of fiona geometries
        :param beta_coef: Bayseg parameter; controls autocorrelation of segments
        :param beta_jump: Bayseg parameter
        :param n_band: How many PCs/bands will be used;
        if set higher than bands actually available all available bands will be used
        :param custom_subsetter: In case not to use all the input bands
        :param data_path_output: Where to save the results?
        :param MMU: Minumum Mapping Unit in km²; below that input will be set as one segment
        :return: saves segmented image to disk using the Bayseg
         # the default should be string_to_raster = tss
        # raster_tsi = string_to_raster.replace('TSS', 'TSI')
        range(0, 35 * 14) for coreg stack
        """
        """
        if os.path.exists(data_path_output + 'output'):
            print('output directory already exists')
            #os.rmdir(data_path_output + 'output')
        else:
            os.mkdir(data_path_output + 'output')
        """
        data_patho = data_path_output + 'output'
        field_counter = "{}{}{}{}{}{}{}".format(str(self.stncl), "_", str(self.beta_jump), "_", str(self.beta_coef),
                                                str(indexo), self._filter)

        three_d_image, two_d_im_pca, mask_local, gt_gdal, MMU_fail = self.prepare_data(string_to_raster, vector_geom)

        if MMU_fail:
            # this will be used when the parcel is smaller than the MMU limit,
            mino = 2
            labels = np.zeros(three_d_image.shape[0] * three_d_image.shape[1])
        elif three_d_image is None:
            return
        elif two_d_im_pca.shape[1] == 0:
            return
        else:

            mino = bayseg.bic(two_d_im_pca, self.n_class)

            ############################################################

            print(three_d_image.shape, two_d_im_pca.shape)
            itero = self.iterations
            clf = bayseg.BaySeg(three_d_image, mino, beta_init=self.beta_coef, stencil=self.stncl, normalize=False)
            clf.fit(itero, beta_jump_length=self.beta_jump)

            # shape: n_iter, flat image, n_classes
            # print('PROBSHAPE: ', prob.shape)
            file_str = "{}{}{}".format(data_patho + "/diagnostics", "_stack_", str(field_counter))
            print(file_str)
            ie = clf.diagnostics_plot(transpose=True, save=True, path_to_save=file_str + '.png', ie_return=True)

            labels = clf.labels[-1, :]
            print('SHAPE LABELS', labels.shape)

        # imageio.mimsave(data_path + 'bayseg.gif', images_iters)
        file_str = "{}{}{}".format(data_patho + "/Bayseg_", str(field_counter), "_")
        # file_str_ie = "{}{}{}".format(data_patho + "/Bayseg_ie_", str(field_counter), "_")
        # to save as integer
        labels_img = np.reshape(labels, (three_d_image.shape[0], three_d_image.shape[1]))

        # ie_img = np.reshape(ie, (three_d_image.shape[0], three_d_image.shape[1])) * 10000
        # prob_img = np.reshape(prob[-1, :, 3], labels_img.shape)

        # labels__img = np.reshape(labels_, (scaled_shaped.shape[0], scaled_shaped.shape[1]))
        labels_img += 1
        labels_img[mask_local] = 0
        labels = labels_img.reshape(three_d_image.shape[0] * three_d_image.shape[1])
        print(three_d_image.shape, labels.shape)
        # labels_img = function(labels_img)
        # plt.imshow(labels_img)
        # plt.show()
        WriteArrayToDisk(labels_img, file_str, gt_gdal, polygonite=True, fieldo=field_counter)
        #
        # WriteArrayToDisk(labels_img, file_str_maj, gt_gdal, polygonite=True, fieldo=field_counter)
        # WriteArrayToDisk(ie_img, file_str_ie, gt_gdal, polygonite=False, fieldo=field_counter)
        if file_str is None:
            print('not returning anything')
        else:
            return file_str + 'polygonized.shp'


def filter_function(invalues):
    invalues_mode = mode(invalues, axis=None, nan_policy='omit')
    return invalues_mode[0]


majority_f = lambda array: generic_filter(array, function=filter_function, size=3)


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


def bilateral_nchannels(array):
    """

    :param array: array dim should be x,y,nchannel
    :return: array with shape x, y, nchannel; CLAHE transformed
    """
    # array dim should be x,y,nchannel
    new_array = np.empty(array.shape)
    for i in range(array.shape[2]):
        array_squeezed = array[:, :, i].squeeze()
        new_array[:, :, i] = cv2.bilateralFilter(array_squeezed, 15, 5, 5)
        # print('clahe', i+1, '/', array.shape[2])
    return new_array


def clahe_nchannels(array):
    """
    :param array: array dim should be x,y,nchannel
    :return: array with shape x, y, nchannel; CLAHE transformed
    """
    # array dim should be x,y,nchannel
    new_array = np.empty(array.shape)
    for i in range(array.shape[2]):
        array_squeezed = array[:, :, i].squeeze()
        new_array[:, :, i] = exposure.equalize_adapthist(array_squeezed, clip_limit=0.01,
                                                         kernel_size=30)  # array.shape[1]/2)
        # print('clahe', i+1, '/', array.shape[2])
    return new_array


