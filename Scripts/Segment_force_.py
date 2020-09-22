import sys
sys.path.append(r'\\141.20.140.91/SAN/_ProjectsII/Grassland/temp/temp_Max/Tools')
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

from joblib import Parallel, delayed
import time
from Wes_Tools.__geometry_tools import *
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *
from affine import Affine


def main():
    prefix_network_drive = r'\\141.20.140.91/SAN/_ProjectsII/Grassland/'
    data_path = prefix_network_drive + 'temp/temp_Max/Data/'

    dp_data = prefix_network_drive + 'SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/'
    vector_paths = Shape_finder(dp_data)
    vector_paths = vector_paths[1:]
    another_counter = 0
    for vector_path in vector_paths:
        force_tile = vector_path.split('/')[-2]

        print(vector_path, force_tile)
        gdf_ = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                                crs="EPSG:3035")

        mask = gdf_.area > 500
        gdf = gdf_.loc[mask]

        box_counter = 100

        PCA_ = [False]
        filter_ = 'bilateral'  # , 'no_filter']
        stencil_ = "4p"  # , "8p"]
        segmentation_rounds = [0.5, 0.01, 0.51]
        n_band = 13
        n_class = 5

        for ct, roundo in enumerate(segmentation_rounds):

            try:
                shutil.rmtree(data_path + 'output/')
            except:
                print('')
            os.mkdir(data_path + 'output')
            print('ROUNd', roundo)


            if roundo == 0.5:
                data_patg_alt = find_matching_raster(vector_path,
                                                    prefix_network_drive + 'SattGruen/Analyse/Mowing_detection/Data/Raster/S-1/',
                                                     ".*[c][k][e][d].*[t][i][f]{1,2}$")
                print(data_patg_alt)
                clf = segmentation_BaySeg(n_band=11, custom_subsetter=range(10, 21), _filter=filter_,
                                          MMU=roundo, into_pca=11, beta_coef=40, beta_jump=1,
                                          PCA=False, n_class=4, iterations=10, neighbourhood=stencil_)
                Parallel(n_jobs=2)(
                    delayed(clf.segment_2)(data_patg_alt, vector_geom=row, data_path_output=data_path,
                                           indexo=index) for index, row in gdf.iterrows())
            if roundo == 0.01:
                different_raster = find_matching_raster(vector_path,
                                                        prefix_network_drive + '/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/',
                                                        ".*[E][V][I].*[B][M].*[t][i][f]{1,2}$")
                # different_raster = r'H:\Grassland\EVI\X0068_Y0042/2017-2019_001-365_HL_TSA_LNDLG_EVI_TSS.tif'
                # different_raster = r'X:\temp\temp_Max/TS_X0068_Y0042.tif'
                clf = segmentation_BaySeg(n_band=n_band, custom_subsetter=range(2, 11), _filter=filter_,
                                          MMU=roundo, into_pca=40, beta_coef=50, beta_jump=1.5,
                                          PCA=False, n_class=n_class, iterations=10)
                Parallel(n_jobs=4)(
                    delayed(clf.segment_2)(different_raster, vector_geom=row, data_path_output=data_path,
                                           indexo=index) for index, row in gdf.iterrows())

            if roundo == 0.51:
                different_raster = find_matching_raster(vector_path,
                                                        prefix_network_drive + '/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/',
                                                        ".*[E][V][I].*[B][M].*[t][i][f]{1,2}$")
                # different_raster = r'H:\Grassland\EVI\X0068_Y0042/2017-2019_001-365_HL_TSA_LNDLG_EVI_TSS.tif'
                # different_raster = r'X:\temp\temp_Max/TS_X0068_Y0042.tif'
                clf = segmentation_BaySeg(n_band=n_band, custom_subsetter=range(2, 11), _filter=filter_,
                                          MMU=roundo, into_pca=40, beta_coef=50, beta_jump=1.5,
                                          PCA=False, n_class=n_class, iterations=10)
                Parallel(n_jobs=4)(
                    delayed(clf.segment_2)(different_raster, vector_geom=row, data_path_output=data_path,
                                           indexo=index) for index, row in gdf.iterrows())

            joined = join_shapes_gpd(data_path + 'output/', own_segmentation='own')
            gdf = joined

            shutil.rmtree(data_path + 'output/')

        field_counter = "{}{}{}{}{}{}{}{}".format(str(filter_), "_", str(stencil_), "_", str(n_band), '_',
                                                  ct, "_" + str(another_counter) + "_")
        box_counter += 1
        another_counter += 10
        print(field_counter)
        segmented_file_path = dp_data + force_tile + '/segmentation_' + force_tile + '.shp'
        joined.to_file(segmented_file_path)


if __name__ == '__main__':
    start = time.time()
    print('started at:', start)
    main()
    end = time.time()
    elapsed_time = end - start
