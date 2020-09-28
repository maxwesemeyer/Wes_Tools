from hubdc.core import *
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import rasterio


def val_func(vector_geom, raster_path):
    shp = [vector_geom.geometry]
    """
    with fiona.open(vector_geom, "r") as shapefile:
        shp = [feature["geometry"] for feature in shapefile]
    """
    print(vector_geom)
    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=0)
        print(out_image)


if __name__ == '__main__':
    reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\mowingEvents.geojson.geojson'
    predicted_raster_sum = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2018\vrt\MowingEvents_SUM_2018.tif'
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(reference_vector)], ignore_index=True),
                           crs="EPSG:3035")

    Parallel(n_jobs=1)(
        delayed(val_func)(reference_vector, predicted_raster_sum) for index, row in gdf.iterrows())

