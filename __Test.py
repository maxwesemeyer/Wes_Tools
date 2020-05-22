import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import fiona
sys.path.append("C:/Users/BorgoDörp/OneDrive/")
from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *

if __name__ == '__main__':
    data_path = 'C:/Users/BorgoDörp/OneDrive/MA_bilder/'
    raster_path = data_path + '2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif'
    list_of_shapes = Shape_finder('O:/Student_Data/Wesemeyer/Master/results_new/')
    with fiona.open(data_path + 'Neu_test.gpkg') as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    print(list_of_shapes)
    pse_list = []
    overall_list = []
    set_global_Cnn_variables(bands=3)
    segment_cnn(raster_path, shapes, data_path=data_path, n_band=3)
    for shapes in list_of_shapes:
        gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(list_of_shapes[23])], ignore_index=True),
                               crs=gpd.read_file(list_of_shapes[23]).crs)

        # drop cluster number 0, which is all no grassland polygons
        indexNames = gdf[gdf['Cluster_nb'] == 0].index
        gdf.drop(indexNames, inplace=True)

        # drop all entries with field nb = na, which don't have a geometry and are duplicates
        indexNames_2 = gdf[np.isnan(gdf['field_nb'])].index
        gdf.drop(indexNames_2, inplace=True)

        x = Parallel(n_jobs=1)(
            delayed(aggregator)(
                raster_NDV='X:/lower_saxony_sentinel2_TSA_coreg/X0061_Y0046/2018-2020_001-365_HL_TSA_SEN2L_NDV_TSS.tif',
                shapei=row, indexo=index, subsetter=None) for
            index, row in gdf.iterrows())
        mergo = pd.DataFrame(x)
        print('Mergo:', mergo)
        try:
            ###
            print('I tried')

        except:
            print('error')
            continue