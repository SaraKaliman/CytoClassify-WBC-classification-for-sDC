# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:49:19 2024

@author: skalima
"""

import dclab
import numpy as np
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = '1'
# sklearn version was 1.5.2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def filter_rtdc(ds, filter_names, min_vals, max_vals):
    bool_ind_selected = np.ones(len(ds["index"]), dtype=bool)
    for feat, min_val, max_val in zip(filter_names, min_vals, max_vals):
        feat_filter = (ds[feat] < max_val) & (ds[feat] > min_val)
        bool_ind_selected = bool_ind_selected & feat_filter
    ds.filter.manual[bool_ind_selected] = False
    ds.filter.manual = np.invert(ds.filter.manual)
    ds.apply_filter()
    ds_wbc = dclab.new_dataset(ds)
    return bool_ind_selected, ds_wbc


def filter_manual(ds, indexes_to_select):
    ds.filter.manual[indexes_to_select] = False
    ds.filter.manual = np.invert(ds.filter.manual)
    ds.apply_filter()
    return dclab.new_dataset(ds)


def gmm_clustering(ds, relevat_feat, n_clussters, cov_type="full"):
    X = np.array([ds[relevat_feat[0]]])
    for feat in relevat_feat[1:]:
        X = np.vstack((X, np.array([ds[feat]])))
    df = pd.DataFrame(X.T, columns=relevat_feat)
    gmm = GaussianMixture(n_components=n_clussters,
                          covariance_type=cov_type,
                          random_state=0).fit(X.T)
    labels = gmm.predict(X.T)
    df["Clusters"] = labels
    df["index"] = np.arange(X.shape[1]) + 1
    return df


def kmeans_clustering(ds, relevat_feat, n_clussters):
    X = np.array([ds[relevat_feat[0]]])
    for feat in relevat_feat[1:]:
        X = np.vstack((X, np.array([ds[feat]])))
    df = pd.DataFrame(X.T, columns=relevat_feat)
    kmeans = KMeans(n_clusters=n_clussters,
                    random_state=0).fit(X.T)
    labels = kmeans.predict(X.T)
    df["Clusters"] = labels
    df["index"] = np.arange(X.shape[1]) + 1
    return df
