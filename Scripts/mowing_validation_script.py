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
                #try:
                #    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d')
                #except:
                #    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d %H:%M:%S')
                pred = single_date_pred
                diff = single_date_ref.date() - pred

                diff_temp_list.append(abs(diff.days))
            out_diff.append(min(diff_temp_list))
        return out_diff
    else:
        print(predicted_dates)
        return []


def compare_dates_new(reference_dates, predicted_dates, year):
    """
    returns the difference in days betweeen the predicted mowing event and the reference mowing event for each predicted;
    The difference with the smallest value is chosen
    :param reference_dates:
    :param predicted_dates:
    :return:
    """

    out_diff = list()
    if len(predicted_dates) >= 1:
        for single_date_pred in predicted_dates:
            pred = single_date_pred
            diff_temp_list = list()
            for single_date_ref in reference_dates:
                if single_date_ref == datetime.datetime.strptime('01.01.18', '%d.%m.%y'):
                    continue
                #try:
                #    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d')
                #except:
                #    pred = datetime.datetime.strptime(str(single_date_pred), '%Y-%m-%d %H:%M:%S')

                diff = single_date_ref.date() - pred

                diff_temp_list.append(abs(diff.days))
            out_diff.append(min(diff_temp_list))
        return out_diff
    else:
        print(predicted_dates)
        return []


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def to_date(doy, yr):
    return date.fromordinal(date(yr, 1, 1).toordinal() + np.int(doy) - 1)


def val_func(vector_geom, raster_path_sum, index, year):
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

    names_list = ["Schnitt_1", "Schnitt_2","Schnitt_3","Schnitt_4","Schnitt_5"]

    for ft, names in zip(vector_geom, vector_geom.index.values):

        if names_list.__contains__(names):
            try:

                ref_mow1 = datetime.datetime.strptime(str(ft), '%d.%m.%Y')
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
        predicted, out_transform = rasterio.mask.mask(src, shp, crop=True, nodata=-9999)
        predicted_sum = predicted[0, :, :]
        predicted_doy = predicted[4:11, :, :]

        predicted_doy = np.reshape(predicted_doy, newshape=(7, predicted_doy.shape[1]*predicted_doy.shape[2]))

        ravelled = predicted.ravel()[np.where(predicted_sum.ravel() != -9999)]
        diff = np.array(list(map(lambda x: x - n_mows, ravelled))).mean()

        # wenn zu viele Predicted = positiv
        # wenn zu wenig predicted = negativ

        #diff = np.int(predicted-n_mows)
        #print(predicted, mows_list[condition], n_mows, diff, index)
        days_diff_list = []
        predicted_list_date = []

        for i in range(predicted_doy.shape[1]):
            ravelled = predicted_doy[:, i]
            if np.any(ravelled != -9999):

                ##############
                ## exlude doy 0, which is the case when no mowing is found
                ravelled = ravelled[np.where(ravelled != 0, True, False)]
                date_list = [to_date(x, year) for x in ravelled]
                if len(date_list) == 0:
                    continue

                days_diff_list.append(np.array(compare_dates_new(mow_dates, date_list, year)))
        print(list(flatten(days_diff_list)))
        for band in range(n_mows):

            ravelled = predicted_doy[band, :].ravel()[np.where(predicted_doy[band, :].ravel() != -9999)]
            ##############
            ## exlude doy 0, which is the case when no mowing is found
            ravelled = ravelled[np.where(ravelled != 0, True, False)]
            predicted_list_date.append(np.median(ravelled))
            #days_diff_list.append(np.array([compare_dates(ref, pred, year) for ref, pred in zip([mow_dates[band]]*len(ravelled), date_list)]).mean())

    return diff, np.nanmean(days_diff_list), predicted_list_date, index


if __name__ == '__main__':
    import glob
    #reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference\mowingEvents_3035.gpkg'
    #reference_vector = r'\\141.20.140.91\SAN\_ProjectsII\Grassland\temp\temp_Marcel\FL_mowing_reference/mowingEvents_3035_out/mowingEvents_3035_out.shp'
    reference_vector = glob.glob(r'X:\temp\temp_Max\Data\Vector\Referenz_Mowing/**shp')
    print(reference_vector)
    predicted_raster = r'H:\Mowing_DC2021\Mosaic\2018/mowingEvents_DC2021_2018.tif'
    #predicted_raster_date = r'\\141.20.140.91\NAS_Rodinia\Croptype\Mowing_2018\vrt\vrt_DOY.vrt'
    gdf = gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in reference_vector], ignore_index=True))

    x = Parallel(n_jobs=1)(
        delayed(val_func)(row, predicted_raster, index, year=2018) for index, row in gdf.iterrows())
    diff_n_mows, diff_days, predicted_dates, index = list(zip(*x))
    d1_list = []
    d2_list = []
    d3_list = []
    d4_list = []
    d5_list = []

    for item in x:

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
    gdf['mean_dev_days'] = pd.Series(diff_days)
    gdf.to_file(r'X:\temp\temp_Max\Data\Vector\VALIDATED.shp')
    print(np.nanmean(np.abs(diff_n_mows)), 'mean deviation n mowings abs')
    print(np.nanmean((diff_n_mows)), 'mean deviation n mowings')

    #ab = itertools.chain(*diff_days)
    #diff_days_array = np.array(list(ab))

    print(diff_days)
    print(np.nanmean(diff_days))