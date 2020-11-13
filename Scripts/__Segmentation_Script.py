
from joblib import Parallel, delayed

import glob
import time
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *
from Wes_Tools.__Join_results import *
from Wes_Tools.__geometry_tools import *
from skopt import BayesSearchCV
import shapely.ops
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon

def main():
    data_path = 'X:/temp/temp_Max/Data/'
    vector_paths = glob.glob(r'X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/' + '*.shp')
    vector_paths = vector_paths[1:]
    vector_paths = ['X:\SattGruen\Analyse\GLSEG\Raster\Vectorized_Alkis/12polygonized.shp']
    vector_paths = ['X:/temp/temp_Max/Data/Vector/Mask_Paulinaue_new.shp']
    #vector_paths = [r'X:\SattGruen\Analyse\GLSEG\Raster\snippets_invekos/stacked_12_9.pngpolygonized.shp']
    another_counter = 0
    ######################################
    # accuracy assessment lists

    pse_list = []
    iou_list = []
    osq_list = []
    overall_list = []
    OS_list = []
    US_list = []
    ed2_list = []
    pse_list = []
    nsr_list = []
    params_list = []
    names = []
    ######################################

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
        #gdf_roads = gpd.GeoDataFrame(pd.concat([gpd.read_file(r'X:\temp\temp_Max\Data\Vector/Paulinaue_roads.gpkg')], ignore_index=True),
        #                        crs="EPSG:3035")

        #gdf_roads['geometry'] = gdf_roads.buffer(10)

        #gdf_ = gpd.overlay(gdf_, gdf_roads, how='difference')

        """
        for line in gdf_roads.geometry:
            out_ = shapely.ops.split(out, line)
            if len(out_) > 1:

                print(len(out))
        """
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
            filter_list = ['bilateral', 'no_filter']
            stencil_list = ["8p"]#, "8p"]
            segmentation_rounds_list = [[0.5, 0.01, 0.00915]]#, [0.5, 0.01, 0.05], [0.5, 0.01, 0.015]]

            gdf_old = gdf
            n_classes_list = [4, 5, 6, 7]
            input_bands_list = [10, 100]
            for n_band in input_bands_list:
                for n_class in n_classes_list:
                    for filter in filter_list:
                        for stncl in stencil_list:

                            for seg_round_counter, segmentation_rounds in enumerate(segmentation_rounds_list):
                                gdf = gdf_old
                                params = (filter, stncl, segmentation_rounds, PCA_, n_class, n_band)
                                for round in segmentation_rounds:
                                    try:
                                        shutil.rmtree(data_path + 'output/')
                                    except:
                                        print('')
                                    os.mkdir(data_path + 'output')
                                    print('ROUNd', round)
                                    # does not work within function with parallel os.mkdir

                                    # set_global_Cnn_variables(bands=par, convs=betas)
                                    # old subsetter range(1,500)
                                    # 104 168 = April - Ende Oktober
                                    # first round 0.05
                                    # second round 0.5

                                    if round == 0.5:
                                        data_patg_alt = find_matching_raster(vector_path,
                                                                             'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/S-1/',
                                                                             ".*[c][k][e][d].*[t][i][f]{1,2}$")
                                        print(data_patg_alt)
                                        clf = segmentation_BaySeg(n_band=11, custom_subsetter=range(10, 21), _filter=filter,
                                                               MMU=round, into_pca=11, beta_coef=40, beta_jump=1,
                                                               PCA=False, n_class=4, iterations=20, neighbourhood=stncl)
                                        Parallel(n_jobs=5)(
                                            delayed(clf.segment_2)(data_patg_alt, vector_geom=row, data_path_output=data_path,
                                                               indexo=index) for index, row in gdf.iterrows())
                                    if round == 0.01:

                                        different_raster = find_matching_raster(vector_path,
                                                                                'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN0_BN0/',
                                                                                ".*[E][V][I].*[S][S].*[t][i][f]{1,2}$")
                                        #different_raster = r'H:\Grassland\EVI\X0068_Y0042/2017-2019_001-365_HL_TSA_LNDLG_EVI_TSS.tif'
                                        #different_raster = r'X:\temp\temp_Max/TS_X0068_Y0042.tif'
                                        clf = segmentation_BaySeg(n_band=100, custom_subsetter=range(2, 11), _filter=filter,
                                                              MMU=round, into_pca=40, beta_coef=50, beta_jump=1.5,
                                                              PCA=False, n_class=3, iterations=20)
                                        Parallel(n_jobs=5)(
                                            delayed(clf.segment_2)(different_raster, vector_geom=row, data_path_output=data_path,
                                                               indexo=index) for index, row in gdf.iterrows())

                                    if round == 0.00915:

                                        different_raster = find_matching_raster(vector_path,
                                                                                'X:/SattGruen/Analyse/Mowing_detection/Data/Raster/AN0_BN0/',
                                                                                ".*[E][V][I].*[S][S].*[t][i][f]{1,2}$")
                                        #different_raster = r'H:\Grassland\EVI\X0068_Y0042/2017-2019_001-365_HL_TSA_LNDLG_EVI_TSS.tif'
                                        #different_raster = r'X:\temp\temp_Max/TS_X0068_Y0042.tif'
                                        clf = segmentation_BaySeg(n_band=n_band, custom_subsetter=range(10, 61), _filter=filter,
                                                              MMU=round, into_pca=40, beta_coef=50, beta_jump=1.5,
                                                              PCA=False, n_class=8, iterations=30)
                                        Parallel(n_jobs=5)(
                                            delayed(clf.segment_2)(different_raster, vector_geom=row, data_path_output=data_path,
                                                               indexo=index) for index, row in gdf.iterrows())
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

                                field_counter = "{}{}{}{}{}{}{}{}".format(str(filter), "_", str(stncl), "_", str(n_band), '_',
                                                                          seg_round_counter, "_" + str(another_counter) + "_")
                                box_counter += 1
                                another_counter += 10
                                print(field_counter)
                                segmented_file_path = data_path + 'joined/seg_gridsrch' + field_counter + '.shp'
                                joined.to_file(segmented_file_path)
                                # pse = Accuracy_Assessment(vector_path, shapes).IoU()

                                reference_path = data_path + 'Vector/Paulienenaue_TF.shp'
                                ref_raster_path = 'X:/SattGruen/Analyse/GLSEG/Raster/X0068_Y0042/2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_TSI.tif'
                                acc_ass = Accuracy_Assessment(reference_path, segmented_file_path, convert_reference=True,
                                                              raster=ref_raster_path)
                                pse, nsr, ed2 = acc_ass.Liu_new()
                                OS, US, Overall = acc_ass.Clinton()
                                iou, osq = acc_ass.IoU()
                                # print((np.array(iou)))
                                print(np.mean(np.array(OS)), np.mean(np.array(US)), np.mean(np.array(Overall)))
                                print('PSE', np.mean(np.array(pse)), np.mean(np.array(nsr)), np.mean(np.array(ed2)), 'OSQ:',
                                      osq)
                                names.append(segmented_file_path)
                                iou_list.append(np.mean(np.array(iou)))
                                osq_list.append(osq)
                                overall_list.append(np.mean(np.array(Overall)))
                                OS_list.append(np.mean(np.array(OS)))
                                US_list.append(np.mean(np.array(US)))

                                ed2_list.append(np.mean(np.array(ed2)))
                                nsr_list.append(np.mean(np.array(nsr)))
                                pse_list.append(np.mean(np.array(pse)))
                                params_list.append(params)

                                dict = {'name': names, 'IoU': iou_list, 'osq': osq_list, 'pse': pse_list,
                                                            'nsr': nsr_list, 'ed2': ed2_list,
                                                            'OS': OS_list, 'US': US_list, 'Overall OS US': overall_list, 'params': params_list}
                                score_frame = pd.DataFrame(dict)
                                score_frame.to_csv(data_path + 'scores_grdshrch.csv')



def aggregate_main(inputshape, data_patg_alt):
    data_path = 'X:/temp/temp_Max/Data/'
    gdf = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(inputshape)], ignore_index=True),
        crs=gpd.read_file(inputshape).crs)
    # drop cluster number 0, which is all no grassland polygons
    #indexNames = gdf[gdf['Cluster_nb'] == 0].index
    #gdf.drop(indexNames, inplace=True)

    # drop all entries with field nb = na, which don't have a geometry and are duplicates
    #indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
    #gdf.drop(indexNames_2, inplace=True)

    x = Parallel(n_jobs=20)(
        delayed(aggregator)(
            raster_NDV=data_patg_alt,
            shapei=row, indexo=index, yr=2018) for
        index, row in gdf.iterrows())
    mergo, dates = list(zip(*x))

    print('mergo', mergo)
    ###########
    # interpolatedf
    year = 2018
    next_year = 2019
    for index, row_mean in enumerate(mergo):


        row_mean = np.array(row_mean)
        x_ = dates[index]
        row_mean = row_mean[:len(x_)]
        year_subsetter = np.where((x_ > year) & (x_ < next_year))
        print(len(x_), len(row_mean))
        row_mean = row_mean[year_subsetter]
        x_ = x_[year_subsetter]

        row_mean[row_mean == 0] = np.nan
        nans, x = nan_helper(row_mean)
        row_mean[nans] = np.interp(x(nans), x(~nans), row_mean[~nans])

        color = np.array([str(1)] * len(nans))
        color[nans] = 'red'
        color[~nans] = 'green'
        plt.scatter(x_, row_mean, s=5, c=color)
        plt.title(str(index))
        plt.savefig(r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference/Plots/' + str(index) + '_2018' + '.png')

        #plt.show()
        plt.close()
    #mergo[mergo.columns[-1]] = mergo[mergo.columns[-1]].astype(dtype=int)
    #merged = gdf.merge(mergo, left_index=True, right_index=False, right_on=mergo[mergo.columns[-1]])
    #print(merged)
    #merged = merged.iloc[:, range(3, 117)]

    # gpd_merged = gpd.GeoDataFrame(merged, crs="EPSG:3035", geometry=merged[0])
    # gpd_merged.to_file(data_path + 'merged_bayseg_raster.shp')
    #merged.to_csv(data_path + 'merged_bayseg_raster.csv')

if __name__ == '__main__':
    reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference/mowingEvents_3035_out/mowingEvents_3035_out.shp'
    data_path = 'X:/temp/temp_Max/Data/'

    #aggregate_main(reference_vector, r'\\141.20.140.91/NAS_Rodinia/Croptype/Grassland/EVI/')

    #df = pd.read_csv(data_path + 'merged_bayseg_raster.csv')
    #df_n = df.iloc[:, range(3, 111)].to_numpy()
    """
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