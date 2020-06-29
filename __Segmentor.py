import sys
from shapely import geometry
from skimage import exposure
from scipy.ndimage.filters import generic_filter
from scipy import ndimage
from scipy.stats import mode
from .__utils import *
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


def segment_2(string_to_raster, vector_geom, indexo=np.random.randint(0, 100000),
              data_path_output=None, beta_coef=1, beta_jump=0.1, n_band=50,
              custom_subsetter=range(0,80),  MMU=0.05, PCA=True, into_pca='all'):
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
    field_counter = "{}{}{}{}{}{}{}".format(str(n_band), "_", str(beta_jump), "_", str(beta_coef), str(indexo), str(PCA))

    three_d_image, two_d_im_pca, mask_local, gt_gdal, MMU_fail = prepare_data(string_to_raster, vector_geom, custom_subsetter,
                                                                    n_band, MMU=MMU, PCA=PCA, into_pca=into_pca)
    if MMU_fail:
        # this will be used when the parcel is smaller than the MMU limit,
        mino = 2
    elif three_d_image is None:
        return
    else:
        n_class = 5
    try:
        mino = bayseg.bic(two_d_im_pca, n_class)
    ############################################################

        print(three_d_image.shape, two_d_im_pca.shape)


        itero = 100

        clf = bayseg.BaySeg(three_d_image, mino, beta_init=beta_coef)
        clf.fit(itero, beta_jump_length=beta_jump)
        clf.diagnostics_plot()
        # shape: n_iter, flat image, n_classes
        # print('PROBSHAPE: ', prob.shape)
        file_str = "{}{}{}".format(data_patho + "/diagnostics", "_stack_", str(field_counter))
        print(file_str)
        ie = clf.diagnostics_plot(transpose=True, save=True, path_to_save=file_str + '.png', ie_return=True)

        labels = clf.labels[-1, :]
    except:
        return

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
    file_str = "{}{}{}".format(data_patho + "/Bayseg_", str(field_counter), "_")
    file_str_ie = "{}{}{}".format(data_patho + "/Bayseg_ie_", str(field_counter), "_")
    # to save as integer
    labels_img = np.reshape(labels, (three_d_image.shape[0], three_d_image.shape[1]))
    ie_img = np.reshape(ie, (three_d_image.shape[0], three_d_image.shape[1])) * 10000
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
    WriteArrayToDisk(ie_img, file_str_ie, gt_gdal, polygonite=False, fieldo=field_counter)
    if file_str is None:
        print('not returning anything')
    else:
        return file_str + 'polygonized.shp'

