import fiona
from shapely.geometry import Polygon, shape
import re
import os
import numpy as np


def Shape_finder(input_path):
    data_path_input = input_path
    file_path_raster = []
    for root, dirs, files in os.walk(data_path_input, topdown=True):
        #dirs[:] = [d for d in dirs if d in folders_BRB]

        for file in files:
            if re.match(".*[s][h][p]{1,2}$", file):
                file_path_raster.append(str(root + file))
            else:
                continue
    return file_path_raster


class Accuracy_Assessment:

    def Clinton(reference_poly, segmentation_poly):
        """
        This functions calculates Oversegmentation, Undersegmentation and Overall accuracy of segmentation according to
        Clinton et al. 2010
        :param reference_poly: path to input shapefile
        :param segmentation_poly: path to input shapefile
        :return: accuracy values Os, Us, Total
        """
        with fiona.open(reference_poly) as shapefile:
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

        segment_counter = 1
        # store values for output
        US_out = []
        OS_out = []
        Overall_out = []
        for shp_seg in shapes_seg:
            #print(feature_list[segment_counter - 1])
            segment_counter += 1
            # save intersect acrea to select the biggest
            intersecz_size = []
            # temp lists
            US_temp = []
            OS_temp = []
            Overall_temp = []

            for shp_ref in shapes_ref:
                shp_seg = shape(shp_seg)
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

            arg_select = np.argmax(np.array(intersecz_size))
            US_out.append(US_temp[arg_select])
            OS_out.append(OS_temp[arg_select])
            Overall_out.append(Overall_temp[arg_select])

        return US_out, OS_out, Overall_out

    def Liu(reference_poly, segmentation_poly):
        """
        Number of Segments Ratio; See Liu et al. 2012 or
        "A review of accuracy assessment for object-based image analysis: From
        per-pixel to per-polygon approaches" by Ye et al. 2018

        :param reference_poly: path to input shapefile
        :param segmentation_poly: path to input shapefile
        :return:
        """
        with fiona.open(reference_poly) as shapefile:
            shapes_ref = [feature["geometry"] for feature in shapefile]
        with fiona.open(segmentation_poly) as shapefile:
            shapes_seg = [feature["geometry"] for feature in shapefile]

        # store values for output
        PSE_list = []

        for shp_seg in shapes_seg:

            # temp lists
            A_seg_list_temp = []
            Area_ref_temp = []
            intersecz_size = []

            for shp_ref in shapes_ref:
                shp_seg = shape(shp_seg)
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
        N_ref = len(shapes_ref)
        N_map = len(shapes_seg)

        NSR_total = abs(N_ref - N_map) / N_ref
        ED2 = np.sqrt((PSE_arr) ** 2 + (NSR_total) ** 2)

        return PSE_arr, NSR_total, ED2

