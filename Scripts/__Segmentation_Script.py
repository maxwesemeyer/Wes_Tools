import sys
from shapely.geometry import Polygon
import shapely
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import os
import fiona
import shutil
import gdal
sys.path.append("X:/temp/temp_Max/")
import glob
import time
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *


def main():
    data_path = 'X:/temp/temp_Max/Data/'
    vector_paths = glob.glob(r'X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/' + '*.shp')
    vector_paths = vector_paths[1:]
    vector_paths = ['X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/12polygonized.shp']
    vector_paths = ['X:/temp/temp_Max/Data/Vector/own_big_mask_3035.shp']
    #vector_paths = [r'X:\SattGruen\Analyse\GLSEG\Raster\snippets_invekos/stacked_12_9.pngpolygonized.shp']
    another_counter = 0
    for vector_path in vector_paths:
        print(vector_path)
        #data_patg_alt = find_matching_raster(vector_path, 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/', ".*[N][D][V].*[B][M].*[t][i][f]{1,2}$")
        #print(data_patg_alt)
        #if data_patg_alt is None:
        #    continue
        #big_box = getRasterExtent(data_patg_alt)
        #boxes = fishnet(big_box, 15000)
        #print(len(boxes))
        gdf_ = gpd.GeoDataFrame(pd.concat([gpd.read_file(vector_path)], ignore_index=True),
                                crs="EPSG:3035")
        mask = gdf_.area > 500
        gdf = gdf_.loc[mask]
        #gdf = gdf_
        #indexNames = gdf[gdf['Cluster_nb'] == 0].index
        #gdf.drop(indexNames, inplace=True)

        box_counter = 100
        boxes = [1]
        for poly in boxes:
            # sub = gpd.GeoDataFrame(gpd.clip(gdf.buffer(0), Polygon(poly).buffer(0.001)))
            covs = 40
            # best set of parameters so far: no PCA, all available bands and Beta=20;
            # according to Liu: no PCA, 11 bands, Beta=100
            # PCA_ = [True, False]
            PCA_ = [False]
            # params_bands = [10, 20, 25]
            params_bands = [11]
            for PC in PCA_:
                for par in params_bands:
                    big_segment_check = False

                    for segmentation_rounds in [0.5]:
                        try:
                            shutil.rmtree(data_path + 'output/')
                        except:
                            print('')
                        os.mkdir(data_path + 'output')
                        print('ROUNd', segmentation_rounds)
                        # does not work within function with parallel os.mkdir

                        # set_global_Cnn_variables(bands=par, convs=betas)
                        # old subsetter range(1,500)
                        # 104 168 = April - Ende Oktober
                        # first round 0.05
                        # second round 0.5
                        if segmentation_rounds == 0.5:
                            data_patg_alt = find_matching_raster(vector_path,
                                                                 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/S-1/',
                                                                 ".*[c][k][e][d].*[t][i][f]{1,2}$")
                            print(data_patg_alt)
                            c = segmentation()
                            Parallel(n_jobs=5)(
                                delayed(segmentation(n_band=par, custom_subsetter=range(10, 21),

                                                   MMU=segmentation_rounds, into_pca=covs, beta_coef=50, beta_jump=1.5,
                                                   PCA=PC, n_class=3, iterations=100).segment_2())(data_patg_alt, vector_geom=row, data_path_output=data_path,
                                                   indexo=index) for index, row in gdf.iterrows())
                        if segmentation_rounds == 0.05 or segmentation_rounds == 0.25:

                            different_raster = find_matching_raster(vector_path,
                                                                    'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN3_BN1/',
                                                                    ".*[N][D][V].*[B][M].*[t][i][f]{1,2}$")
                            different_raster = find_matching_raster(vector_path,
                                                                 'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/S-1/',
                                                                 ".*[c][k][e][d].*[t][i][f]{1,2}$")

                            segment_2(different_raster, vector_geom=vector_path, data_path_output=data_path,
                                                   indexo=200, n_band=11, custom_subsetter=range(10, 21),
                                                   # custom_subsetter=range(1, 4*12), #custom_subsetter=range(1, 300),# #custom_subsetter=range(1,392),
                                                   MMU=segmentation_rounds, into_pca=10, PCA=False, #beta_coef=60, beta_jump=1,PCA=False, n_class=6, iterations=100
                                                   convs=5,  lr_var=0.1)
                            """
                            Parallel(n_jobs=1)(
                                delayed(segment_cnn)(r'X:\temp\temp_Max\Data\Train_data\train\image/stacked_1_0.tif',
                                                     r'X:\temp\temp_Max\Data\Train_data\train\label/stacked_1_0.png', vector_geom=row, data_path_output=data_path,
                                                   indexo=index, n_band=3, custom_subsetter=range(10, 21),
                                                   # custom_subsetter=range(1, 4*12), #custom_subsetter=range(1, 300),# #custom_subsetter=range(1,392),
                                                   MMU=segmentation_rounds, into_pca=10, PCA=True, #beta_coef=60, beta_jump=1,PCA=False, n_class=6, iterations=100
                                                   convs=3) for index, row in gdf.iterrows())
                            """

                        if not os.listdir(data_path + 'output/'):
                            print('directory empty')
                        else:
                            joined = join_shapes_gpd(data_path + 'output/', own_segmentation='own')
                            gdf = joined
                        if os.path.exists(data_path + 'joined'):
                            print('output directory already exists')

                        else:
                            os.mkdir(data_path + 'joined')
                        print('joined data frame:', joined)
                        shutil.rmtree(data_path + 'output/')

                    field_counter = "{}{}{}{}{}{}{}{}".format(str(PC), "_", str(par), "_", str(covs), '_', box_counter,
                                                              "_" + str(another_counter) + "_")
                    box_counter += 1
                    another_counter += 10
                    print(field_counter)
                    joined.to_file(data_path + 'joined/threetepSeg_s1_s2_FBM_' + field_counter + '.shp')


def aggregate_main(inputshape, data_patg_alt):
    data_path = 'X:/temp/temp_Max/Data/'
    gdf = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(inputshape)], ignore_index=True),
        crs=gpd.read_file(inputshape).crs)
    # drop cluster number 0, which is all no grassland polygons
    indexNames = gdf[gdf['Cluster_nb'] == 0].index
    gdf.drop(indexNames, inplace=True)

    # drop all entries with field nb = na, which don't have a geometry and are duplicates
    indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
    gdf.drop(indexNames_2, inplace=True)

    x = Parallel(n_jobs=20)(
        delayed(aggregator)(
            raster_NDV=data_patg_alt,
            shapei=row, indexo=index, subsetter=range(1, 57)) for
        index, row in gdf.iterrows())

    mergo = pd.DataFrame(x)
    print(list(mergo.columns.values))

    mergo[mergo.columns[-1]] = mergo[mergo.columns[-1]].astype(dtype=int)
    merged = gdf.merge(mergo, left_index=True, right_index=False, right_on=mergo[mergo.columns[-1]])
    print(merged)
    merged = merged.iloc[:, range(3, 117)]

    # gpd_merged = gpd.GeoDataFrame(merged, crs="EPSG:3035", geometry=merged[0])
    # gpd_merged.to_file(data_path + 'merged_bayseg_raster.shp')
    merged.to_csv(data_path + 'merged_bayseg_raster.csv')

if __name__ == '__main__':
    """
    data_path = 'X:/temp/temp_Max/Data/'
    #aggregate_main(r'X:\temp\temp_Max\Data\joined\threetepSeg_s1_s2_new_True_2_40_100_0_.shp',
    #               r'X:\temp\temp_Marcel\S-1_test\Ramin\Ramin_S1_Coh_resample_10.dat')
    df = pd.read_csv(data_path + 'merged_bayseg_raster.csv')
    df_n = df.iloc[:, range(3, 111)].to_numpy()
    from sklearn.cluster import AgglomerativeClustering
    print(df_n.shape)
    labels = AgglomerativeClustering(n_clusters=5).fit_predict(df_n)
    print(labels.shape)
    df['clusters'] = labels
    df.to_csv(data_path + 'merged_bayseg_raster_labels.csv')
    """
    start = time.time()
    print('started at:', start)
    main()
    end = time.time()
    elapsed_time = end - start





"""
IDEAs: 

define/find areas of no/low regrowth; the VI values in these areas can then be adapted
first idea: find points below a NDVI value of ~0.5 then see if regrowth or not
unsupervised video segmentation? 

TODO:
Sentinel 1 für Paulinaue; Spectral Temporal metrics für paulinaue; Mit Grünlandmaske für Paulinaue
Für mowing detection: gesamtes gebiet mit spectral temproal metrics; Coherence; combined; PCA

"""