import sys
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import fiona
sys.path.append("X:/temp/temp_Max/")

from Wes_Tools.Accuracy_ import *
from Wes_Tools.Plots_OBIA import *
from Wes_Tools.__Segmentor import *
from Wes_Tools.__CNN_segment import *


if __name__ == '__main__':
    data_path = 'X:/temp/temp_Max/Data/'
    data_patg_alt = 'X:/SattGruen/Analyse/GLSEG/Raster'
    raster_path = 'X:/SattGruen/Analyse/GLSEG/Raster/X0068_Y0043/2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_TSS.tif'
    list_of_raster = Tif_finder(data_patg_alt)
    print(list_of_raster)
    list_of_shapes = Shape_finder(data_path + 'Polygon_Ribbek/')
    print(list_of_shapes)
    set_global_Cnn_variables(bands=3)

    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(list_of_shapes[1])], ignore_index=True),
                           crs=gpd.read_file(list_of_shapes[1]).crs)
    """
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
    """
    Parallel(n_jobs=1)(delayed(segment_2)(raster_path, vector_geom=row, data_path_output=data_path,
                                            indexo=index, n_band=3, custom_subsetter=range(5,60)) for index, row in gdf.iterrows())

