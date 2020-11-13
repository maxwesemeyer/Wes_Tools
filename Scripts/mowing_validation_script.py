from hubdc.core import *
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
import rasterio.mask
from datetime import date
import itertools


def compare_dates(reference_dates, predicted_dates, year):
    """
    returns the difference in days betweeen the predicted mowing event and the reference mowing event;
    The difference with the smallest value is chosen
    :param reference_dates:
    :param predicted_dates:
    :return:
    """

    out_diff = list()
    if len(predicted_dates) >= 1:
        for single_date_ref in reference_dates:
            if single_date_ref == datetime.datetime.strptime('01.01.18', '%d.%m.%y'):
                continue
            diff_temp_list = list()
            for single_date_pred in predicted_dates:
                try:
                    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d')
                except:
                    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d %H:%M:%S')
                diff = single_date_ref - pred

                diff_temp_list.append(abs(diff.days))
            out_diff.append(min(diff_temp_list))
        return out_diff
    else:
        print(predicted_dates)
        return []


def val_func(vector_geom, raster_path_sum, raster_path_date, index, year):
    shp = [vector_geom.geometry]
    """
    with fiona.open(vector_geom, "r") as shapefile:
        shp = [feature["geometry"] for feature in shapefile]
    """
    begin_2017 = datetime.datetime.strptime(str('2017-01-01'), '%Y-%m-%d')
    begin_2018 = datetime.datetime.strptime(str('2018-01-01'), '%Y-%m-%d')
    begin_2019 = datetime.datetime.strptime(str('2019-01-01'), '%Y-%m-%d')
    begin_2020 = datetime.datetime.strptime(str('2020-01-01'), '%Y-%m-%d')
    mows_list = []
    for ft in vector_geom:
        if ft:
            try:
                ref_mow1 = datetime.datetime.strptime(str(ft), '%Y-%m-%d')
                mows_list.append(ref_mow1)
            except:
                continue
    mows_list = np.array(mows_list)
    if year == 2017:
        condition = np.where((begin_2017 < mows_list) & (mows_list < begin_2018))
    elif year == 2018:
        condition = np.where((begin_2018 < mows_list) & (mows_list < begin_2019))
    elif year == 2019:
        condition = np.where((begin_2019 < mows_list) & (mows_list < begin_2020))
    mow_dates = mows_list[condition]
    n_mows = len(mows_list[condition])
    with rasterio.open(raster_path_sum) as src:
        predicted, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=0)
        # wenn zu viele Predicted = positiv
        # wenn zu wenig predicted = negativ

        diff = np.int(predicted-n_mows)
        print(predicted, mows_list[condition], n_mows, diff, index)
    with rasterio.open(raster_path_date) as src:
        out_image, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=0)
        # doy to date list
        out_list = list(out_image)
        predicted_list_date = []
        for item in out_list:
            if np.int(item) == 0:
                continue
            d1 = date.fromordinal(date(year, 1, 1).toordinal() + np.int(item) - 1)
            predicted_list_date.append(d1)

        days_diff = compare_dates(mow_dates, predicted_list_date, year)
    return diff, days_diff, predicted_list_date, index


if __name__ == '__main__':
    reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference\mowingEvents_3035.gpkg'
    reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference/mowingEvents_3035_out/mowingEvents_3035_out.shp'
    predicted_raster_sum = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2018\vrt\MowingEvents_SUM_2018.tif'
    predicted_raster_date = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2018\vrt\vrt_DOY.vrt'
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(reference_vector)], ignore_index=True))

    x = Parallel(n_jobs=10)(
        delayed(val_func)(row, predicted_raster_sum, predicted_raster_date, index, year=2018) for index, row in gdf.iterrows())
    diff_n_mows, diff_days, predicted_dates, index = list(zip(*x))
    d1_list = []
    d2_list = []
    d3_list = []
    d4_list = []
    d5_list = []

    for item in x:
        print(item)
        for ix in range(5):
            dates = item[2]
            if ix == 0:
                try:
                    d1_list.append(str(dates[0]))
                except:
                    d1_list.append('NULL')
            elif ix == 1:
                try:
                    d2_list.append(str(dates[1]))
                except:
                    d2_list.append('NULL')

            elif ix == 2:
                try:
                    d3_list.append(str(dates[2]))
                except:
                    d3_list.append('NULL')

            elif ix == 3:
                try:
                    d4_list.append(str(dates[3]))
                except:
                    d4_list.append('NULL')

            elif ix == 4:
                try:
                    d5_list.append(str(dates[4]))
                except:
                    d5_list.append('NULL')

    print(d1_list, 'D!LIST')
    series = pd.Series(diff_n_mows)
    gdf['diff_18'] = series
    gdf['cut1_pr_18'] = pd.Series(d1_list)
    gdf['cut2_pr_18'] = pd.Series(d2_list)
    gdf['cut3_pr_18'] = pd.Series(d3_list)
    gdf['cut4_pr_18'] = pd.Series(d4_list)
    gdf['cut5_pr_18'] = pd.Series(d5_list)

    gdf.to_file(r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference\mowingEvents_3035_out')
    print(np.mean(np.abs(diff_n_mows)), 'mean deviation n mowings abs')
    print(np.mean((diff_n_mows)), 'mean deviation n mowings')

    ab = itertools.chain(*diff_days)
    diff_days_array = np.array(list(ab))

    print(diff_days_array)
    print(diff_days_array.mean())
