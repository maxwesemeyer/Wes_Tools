import fiona
from shapely.geometry import Polygon, shape
import numpy as np
import rasterio
from .__utils import *
from .__Join_results import *
import matplotlib.pyplot as plt
from affine import Affine
import shutil
import os




def adapt_to_pixels(reference_poly, raster):
    if os.path.exists('X:/temp/temp_Max/Data/temp/'):
        return
    else:
        i = 0
        os.mkdir('X:/temp/temp_Max/Data/temp/')
        for shp in reference_poly:
            with rasterio.open(raster) as src:
                out_image, out_transform = rasterio.mask.mask(src, [shp], crop=True, nodata=0)
                gt_gdal = Affine.to_gdal(out_transform)
                mask = create_mask_from_ndim(out_image)
                WriteArrayToDisk(mask, 'X:/temp/temp_Max/Data/temp/' + str(i), gt_gdal,
                                 polygonite=True, fieldo=i, EPSG=3035)
                i += 1
        joined = join_shapes_gpd('X:/temp/temp_Max/Data/temp/', own_segmentation='own')
        joined.to_file('X:/temp/temp_Max/Data/temp/adatpted_to_raster.shp')
        #shutil.rmtree('X:/temp/temp_Max/Data/temp/')
        return joined




def accuracy_prep(reference_poly, segmentation_poly, convert_reference=False, raster=None):
    with fiona.open(reference_poly) as shapefile:
        shapes_ref = [feature["geometry"] for feature in shapefile]
        if convert_reference:
            """
            eg. convert reference polygons to Sentinel pixel size
            """
            adapt_to_pixels(shapes_ref, raster)
            with fiona.open('X:/temp/temp_Max/Data/temp/adatpted_to_raster.shp') as shapefile:
                shapes_ref = [feature["geometry"] for feature in shapefile]

    with fiona.open(segmentation_poly) as shapefile:
        shapes_seg = []
        feature_list = []
        for features in shapefile:
            if features["properties"]['Cluster_nb'] != 0 and features["properties"]['field_nb'] != None:
                shapes_seg.append(features["geometry"])
                values_view = features["properties"].values()
                value_iterator = iter(values_view)
                first_value = next(value_iterator)
                feature_list.append(first_value)
            else:
                continue
    return shapes_ref, shapes_seg, feature_list


class Accuracy_Assessment:

    def __init__(self, reference_poly, segmentation_poly, convert_reference=False, raster=None):
        self.shapes_ref, self.shapes_seg, self.feature_list = accuracy_prep(reference_poly, segmentation_poly,
                                                                            convert_reference, raster)

    def Clinton(self):
        """
        This functions calculates Oversegmentation, Undersegmentation and Overall accuracy of segmentation according to
        Clinton et al. 2010
        :param reference_poly: path to input shapefile
        :param segmentation_poly: path to input shapefile
        :return: accuracy values Os, Us, Total
        """

        segment_counter = 1
        # store values for output
        US_out = []
        OS_out = []
        Overall_out = []
        for shp_seg in self.shapes_seg:
            #print(feature_list[segment_counter - 1])
            segment_counter += 1
            # save intersect acrea to select the biggest
            intersecz_size = []
            # temp lists
            US_temp = []
            OS_temp = []
            Overall_temp = []

            for shp_ref in self.shapes_ref:
                # buffer with zero distance to avoid self intersection error
                shp_seg = shape(shp_seg).buffer(distance=0)
                A_int = shp_seg.intersection(shape(shp_ref)).area

                if A_int == 0:
                    continue
                else:
                    intersecz_size.append(A_int)
                    A_ref = shape(shp_ref).area
                    A_map = shp_seg.area

                    US = 1 - A_int/A_ref
                    OS = 1 - A_int/A_map
                    Overall = np.sqrt( ((US)**2 + (OS)**2)/2)
                    US_temp.append(US)
                    OS_temp.append(OS)
                    Overall_temp.append(Overall)
            if np.any(np.array(intersecz_size) > 1):
                arg_select = np.argmax(np.array(intersecz_size))
                US_out.append(US_temp[arg_select])
                OS_out.append(OS_temp[arg_select])
                Overall_out.append(Overall_temp[arg_select])
            else:
                US_out.append(1)
                OS_out.append(1)
                Overall_out.append(1)
        return US_out, OS_out, Overall_out

    def Liu(self):
        """
        Number of Segments Ratio; See Liu et al. 2012 or
        "A review of accuracy assessment for object-based image analysis: From
        per-pixel to per-polygon approaches" by Ye et al. 2018

        :param reference_poly: path to input shapefile
        :param segmentation_poly: path to input shapefile
        :return:
        """

        # store values for output
        PSE_list = []

        for shp_seg in self.shapes_seg:
            # temp lists
            A_seg_list_temp = []
            Area_ref_temp = []
            intersecz_size = []

            for shp_ref in self.shapes_ref:
                # buffer with zero distance to avoid self intersection error
                shp_seg = shape(shp_seg).buffer(distance=0)
                A_int = shp_seg.intersection(shape(shp_ref)).area
                A_ref = shape(shp_ref).area
                A_seg = shp_seg.area

                # areal_overlap_based_criteria =
                # the area of intersection between a reference polygon and the candidate segment is more than half the area of
                # either the reference polygon or the candidate segment
                if A_int == 0:
                    continue
                elif A_int > A_ref / 2 or A_int > A_seg / 2:

                    intersecz_size.append(A_int)
                    A_seg_list_temp.append(A_seg)
                    Area_ref_temp.append(A_ref)

                else:
                    # print(A_int, A_ref / 2, 'second condition', A_int , A_seg / 2)
                    continue
            if np.any(np.array(intersecz_size) > 1):
                PSE = abs(np.sum(np.array(A_seg_list_temp) - np.array(Area_ref_temp))) / np.sum(Area_ref_temp)
                PSE_list.append(PSE)
            else:
                # assuming max error
                PSE_list.append(1)

        PSE_arr = np.array(PSE_list)
        N_ref = len(self.shapes_ref)
        N_map = len(self.shapes_seg)

        NSR_total = abs(N_ref - N_map) / N_ref
        ED2 = np.sqrt((PSE_arr) ** 2 + (NSR_total) ** 2)

        return PSE_arr, NSR_total, ED2


    def IoU(self):
        """
        :param reference_poly: path to input shapefile
        :param segmentation_poly: path to input shapefile
        :return:
        """

        # store values for output
        IUC_list = []
        feature_counter = 0
        for shp_seg in self.shapes_seg:
            # temp lists
            feature_counter += 1
            Union_temp = []
            intersecz_size = []

            for shp_ref in self.shapes_ref:
                # buffer with zero distance to avoid self intersection error
                shp_seg = shape(shp_seg).buffer(distance=0)
                A_int = shp_seg.intersection(shape(shp_ref)).area
                A_un = shp_seg.union(shape(shp_ref)).area
                if A_int == 0:
                    continue
                else:
                    Union_temp.append(A_un)
                    intersecz_size.append(A_int)
            if np.any(np.array(intersecz_size) > 1):

                index = np.argmax(np.array(intersecz_size))
                IUC = np.array(intersecz_size[index])/np.array(Union_temp)[index]
                IUC_list.append(IUC)
            else:
                # assuming max error
                IUC_list.append(0)
        IUC_arr = np.array(IUC_list)
        return IUC_arr

