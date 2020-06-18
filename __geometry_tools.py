from shapely.geometry import box
import gdal


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