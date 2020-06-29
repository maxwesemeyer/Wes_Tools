from shapely.geometry import box
from shapely.wkb import loads
import gdal
import ogr
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *


def fishnet(geometry, threshold):
    bounds = geometry.bounds
    xmin = bounds[0]
    xmax = bounds[2]
    ymin = bounds[1]
    ymax = bounds[3]
    n = int((xmax-xmin)/threshold)
    print('splitting will result in', n**2, ' polygons')
    ncols = int(xmax - xmin + 1)
    nrows = int(ymax - ymin + 1)
    result = []
    for i in range(0, (n)*threshold, threshold):
        for j in range(0, (n)*threshold, threshold):
            b = box(xmin+j, ymin+i, xmin+j+threshold, ymin+i+threshold)
            result.append(b)
    return result


def getRasterExtent(raster_path):
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    projections = []

    raster = gdal.Open(raster_path)
    projections.append(raster.GetProjection())
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    f_xmin, f_pwidth, f_xskew, f_ymax, f_yskew, f_pheight = raster.GetGeoTransform()
    xmins.append(f_xmin)
    xmaxs.append(f_xmin + (f_pwidth * cols))
    ymaxs.append(f_ymax)
    ymins.append(f_ymax + (f_pheight * rows))
    del raster

    x_min = max(xmins)
    y_min = max(ymins)
    x_max = min(xmaxs)
    y_max = min(ymaxs)
    UL = [x_min, y_max]
    LR = [x_max, y_min]

    """
    shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)
    """
    return box(x_min, y_min, x_max, y_max)


def find_matching_raster(vector_path, raster_path, search_string):
    raster_paths = Tif_finder(raster_path, search_string)
    print(raster_paths)
    match = False
    i = 0

    while not match:
        extent_ = getRasterExtent(raster_paths[i])
        file = ogr.Open(vector_path)
        shape = file.GetLayer(0)
        feature = shape.GetFeature(0)
        geom = loads(feature.GetGeometryRef().ExportToWkb())

        if geom.within(extent_):
            match = True
            return raster_paths[i]
        elif i == len(raster_paths)-1:
            print('no matching raster found for', vector_path)
            return None
        else:
            i += 1