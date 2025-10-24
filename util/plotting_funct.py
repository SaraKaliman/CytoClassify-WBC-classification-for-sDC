# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:52:36 2024

@author: skalima
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_defor_area(ds_wbc, cmap_col, plot_num):
    if len(ds_wbc["area_um_raw"][:]) < 20:
        return
    df = pd.DataFrame()
    df["area_um_raw"] = ds_wbc["area_um_raw"][:]
    df["deform_raw"] = ds_wbc["deform_raw"][:]

    # Plot WBC after just standard filters
    values = np.vstack([df["area_um_raw"], df["deform_raw"]])
    kernel = stats.gaussian_kde(values)(values)
    sns.scatterplot(data=df,
                    x='area_um_raw', y='deform_raw',
                    c=kernel,
                    cmap=cmap_col
                    )
    if plot_num:
        x_pos = df["area_um_raw"].min()
        y_pos = 0.3
        plt.text(x_pos, y_pos, "N=" + str(df.shape[0]),
                 horizontalalignment='left',
                 size='large',
                 color='black')


def plot_cnvx_defor_area(ds_wbc, cmap_col, plot_num):
    if len(ds_wbc["area_um"][:]) < 20:
        return
    df = pd.DataFrame()
    df["area_um"] = ds_wbc["area_um"][:]
    df["deform"] = ds_wbc["deform"][:]

    # Plot WBC after just standard filters
    values = np.vstack([df["area_um"], df["deform"]])
    kernel = stats.gaussian_kde(values)(values)
    sns.scatterplot(data=df,
                    x='area_um', y='deform',
                    c=kernel,
                    cmap=cmap_col
                    )
    if plot_num:
        x_pos = df["area_um"].min()
        y_pos = 0.3
        plt.text(x_pos, y_pos, "N=" + str(df.shape[0]),
                 horizontalalignment='left',
                 size='large',
                 color='black')


def plot_time_and_feature(ds, feat):
    max_N = 4*10**4
    N = len(ds["time"])
    if N < 10:
        return
    df = pd.DataFrame()
    if N > max_N:
        sample = np.random.randint(low=0, high=N, size=max_N)
        df["time"] = ds["time"][sample]
        df[feat] = ds[feat][sample]
    else:
        df["time"] = ds["time"][:]
        df[feat] = ds[feat][:]

    # Plot feature as a function of time
    values = np.vstack([df["time"], df[feat]])
    kernel = stats.gaussian_kde(values)(values)
    sns.scatterplot(data=df,
                    x='time', y=feat,
                    c=kernel,
                    cmap="viridis")
    plt.plot([df["time"].min(), df["time"].max()],
             [df[feat].mean(), df[feat].mean()], color="k")


def plot_line_for_time(ds, feat):
    N = len(ds["time"])
    if N < 10:
        return
    df = pd.DataFrame()
    df["time"] = np.floor(ds["time"][:] / 20) * 20
    df[feat] = ds[feat][:]
    sns.lineplot(data=df, x="time", y=feat)
