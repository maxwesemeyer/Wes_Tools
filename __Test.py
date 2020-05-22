sys.path.append("C:/Users/BorgoDörp/OneDrive/My_OBIA_package/")
from wesobia.Accuracy_ import *
from wesobia.Plots_OBIA import *
from wesobia.__Segmentor import *

if __name__ == '__main__':
    data_path = 'C:/Users/BorgoDörp/OneDrive/MA_bilder/'
    raster_path = data_path + '2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif'
    list_of_shapes = Shape_finder('W:/Student_Data/Wesemeyer/Master/results_new/')
    print(list_of_shapes)
    pse_list = []
    overall_list = []

    for shapes in list_of_shapes:

        try:
            gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(list_of_shapes[23])], ignore_index=True),
                                   crs=gpd.read_file(list_of_shapes[23]).crs)

            # drop cluster number 0, which is all no grassland polygons
            indexNames = gdf[gdf['Cluster_nb'] == 0].index
            gdf.drop(indexNames, inplace=True)

            # drop all entries with field nb = na, which don't have a geometry and are duplicates
            indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
            gdf.drop(indexNames_2, inplace=True)

            x = Parallel(n_jobs=1)(
                delayed(aggregator)(raster_NDV='X:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif', shapei=row, indexo=index, subsetter=None) for
                index, row in gdf.iterrows())
            mergo = pd.DataFrame(x)
            print('Mergo:', mergo)
        except:
            print('error')
            continue