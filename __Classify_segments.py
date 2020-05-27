import pandas as pd
import geopandas as gpd
from sklearn.mixture import BayesianGaussianMixture


def classify_unsupervised(pandas_df, ncluster, drop=True):
    if drop:
        parcel_frame = pandas_df.drop(['Unnamed: 0', '73', '72', 'field_nb', 'Cluster_nb'], 1)
    else:
        parcel_frame = pandas_df
    print(parcel_frame)
    parcel_array = parcel_frame.to_numpy()
    labels = BayesianGaussianMixture(n_components=ncluster, covariance_type='full',
                                     max_iter=500, reg_covar=0.1, weight_concentration_prior=0.1, n_init=100,
                                     warm_start=True).fit_predict(parcel_array)
    print(labels.shape)
    return labels


def classify_supervised(pandas_df, a_trained_model, drop=True):
    if drop:
        parcel_frame = pandas_df.drop(['Unnamed: 0', '73', '72', 'field_nb', 'Cluster_nb'], 1)
    else:
        parcel_frame = pandas_df
    print(parcel_frame)
    parcel_array = parcel_frame.to_numpy()
    labels = a_trained_model.predict(parcel_array)
    print(labels.shape)
    return labels
