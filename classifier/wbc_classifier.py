# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:06:55 2024

@author: skalima
"""

import dclab
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util.filter_and_clustering_functions import filter_rtdc
from util.filter_and_clustering_functions import filter_manual
from util.filter_and_clustering_functions import gmm_clustering
from util.filter_and_clustering_functions import kmeans_clustering
from util.plotting_funct import plot_defor_area
from util.plotting_funct import plot_cnvx_defor_area
from util.plotting_funct import plot_time_and_feature
from util.plotting_funct import plot_line_for_time


class wbc_classifier:

    def __init__(self, rtdc_files_folder, file_name,
                 output_folder,
                 export_fname,
                 out_scalar_feat,
                 add_basin_feat,
                 df_count=None):

        self.fname = file_name
        self.output_folder = output_folder
        self.export_fname = export_fname
        self.scalar_feat = out_scalar_feat
        self.df_count = pd.DataFrame(columns=["Leukocytes",
                                              "Lymphocytes",
                                              "Granulo-Monocytes",
                                              "Neutrophils",
                                              "Monocytes",
                                              "Eosinophils"],
                                     index=["Count",
                                            "Percent"])
        self.perceed = True

    def filter_wbc(self):
        print("WBC filering started")
        fname = "wbc_" + self.export_fname
        fpath = Path(self.output_folder) / fname
        excel_file_name = "wbc_count_table_" + \
            self.export_fname.split(".")[0] + ".xlsx"
        excel_fpath = Path(self.output_folder) / excel_file_name
        if fpath.is_file() and excel_fpath.is_file():
            print(fname, "already exists in", self.output_folder)
            print("To recalculated clean and unclean WBC remove files:")
            print(fname, excel_file_name)
            self.df_count = pd.read_excel(
                self.output_folder / excel_file_name,
                index_col=0)
            return

        ds, new_features, good_rtdc = read_main_rtdc_file(self.fname,
                                                          self.scalar_feat)
        self.perceed = good_rtdc
        if not self.perceed:
            return
        self.new_features = new_features

        # wbc are selected using standard filters
        boolian_indexes, ds_wbc = filter_rtdc(ds,
                                              filter_names=["area_um",
                                                            "deform",
                                                            "area_ratio"],
                                              min_vals=[30, 0, 1],
                                              max_vals=[100, 0.12, 1.1])

        if len(ds_wbc["area_um"]) < 500:
            print("Too little WBC are found.")
            self.perceed = False
            return

        indexes_selected = np.where(boolian_indexes == True)[0]
        print("Step1: WBC box-filters is done.")
        # save wbc
        dclab.set_temporary_feature(rtdc_ds=ds_wbc,
                                    feature="userdef0",
                                    data=indexes_selected)
        file_name = "Step1_wbc_unclean_" + self.export_fname
        ds_wbc.export.hdf5(path=self.output_folder / file_name,
                           filtered=True,
                           features=self.new_features,
                           override=True)
        print("WBC-file saved as", file_name)

        # ---- plot kde deform-area for filters for wbc
        plt.figure()
        plot_cnvx_defor_area(ds_wbc, cmap_col="viridis", plot_num=False)
        plt.title("area, deform, porosity filters")
        plt.savefig(self.output_folder / "figures" / "Step1 WBC.jpg",
                    dpi=250)

        # cleaning wbc from rbc doublets
        if len(ds_wbc["area_um"]) < 300:
            self.perceed = False
            return

        # GMM for RBC-doublets
        # Texture First measure of correlation (ptp)
        relevat_feat = ["tex_idm_ptp",
                        "tex_den_ptp"]
        df = gmm_clustering(ds_wbc, relevat_feat, 2)
        plt.figure()
        sns.scatterplot(data=df, x=relevat_feat[0], y=relevat_feat[1],
                        hue="Clusters")
        plt.title("Cleaning WBC from RBC doublets")
        plt.savefig(self.output_folder / "figures" / "Step2 clustering rbc-doublets WBC.jpg",
                    dpi=250)
        # recognize wbc cluster
        # left_cluster = (df["tex_idm_ptp"] == df["tex_idm_ptp"].min()).values
        left_cluster = (df["tex_idm_ptp"] < np.percentile(
            df["tex_idm_ptp"], 50)).values
        wbc_label = int(df["Clusters"][left_cluster].median())
        subset_indexes_to_select = (df["Clusters"] == wbc_label).values
        ds_wbc = filter_manual(ds_wbc, subset_indexes_to_select)
        wbc_indexes = indexes_selected[subset_indexes_to_select]
        print("Step2: WBC cleaning from RBC-doublets done.")

        if len(ds_wbc["area_um"]) < 300:
            self.perceed = False
            return

        # Filtering out stiff rbc and some micro-clots:
        boolian_indexes, ds_wbc = filter_rtdc(ds_wbc,
                                              filter_names=["bright_perc_90",
                                                            "deform_raw"],
                                              min_vals=[-10, 0.06],
                                              max_vals=[15, 0.26])
        wbc_indexes = wbc_indexes[boolian_indexes]
        print("Step2: WBC box-filters for stiff RBC and micro-clots done.")

        # save wbc
        dclab.set_temporary_feature(rtdc_ds=ds_wbc,
                                    feature="userdef0",
                                    data=wbc_indexes)
        file_name = "wbc_" + self.export_fname
        ds_wbc.export.hdf5(path=self.output_folder / file_name,
                           filtered=True,
                           features=self.new_features,
                           override=True)
        print("WBC-file saved as", file_name)

        # ----- plot kde area-deform for WBC
        plt.figure()
        plot_cnvx_defor_area(ds_wbc, cmap_col="viridis", plot_num=True)
        plt.title("Selected WBC")
        plt.savefig(self.output_folder / "figures" / "Step 2 selected WBC.jpg",
                    dpi=250)

        plt.close('all')
        ds.close()
        ds_wbc.close()

    def cluster_wbc(self):
        save_unclean_lym = True
        print("WBC clustering started")
        if self.perceed:
            # read WBC file
            file_name = "wbc_" + self.export_fname
            ds_wbc = dclab.new_dataset(self.output_folder / file_name)
            wbc_indexes = ds_wbc["userdef0"][:]
            self.new_features = list(ds_wbc.features_innate)
        else:
            return

        if len(wbc_indexes) < 300:
            self.perceed = False
            return

        # breaking WBC in Lym and GM based on raw area, deform and texture
        relevat_feat = ["area_um", "area_um_raw"]
        df = kmeans_clustering(ds_wbc, relevat_feat, 2)
        df["deform_raw"] = ds_wbc["deform_raw"][:]
        plt.figure()
        sns.scatterplot(data=df, x="area_um_raw", y="deform_raw",
                        hue="Clusters")
        plt.title("Step3: Seperating lyphocytes")
        plt.savefig(self.output_folder / "figures" / "Step3 seperating lyphocytes.jpg",
                    dpi=250)

        # Save GM cluster
        right_cluster = (df["area_um"] > 60).values
        gm_label = df["Clusters"][right_cluster].median()
        subset_indexes_to_select = (df["Clusters"] == gm_label).values
        ds_gm = filter_manual(ds_wbc, subset_indexes_to_select)
        gm_indexes = wbc_indexes[subset_indexes_to_select]

        # ----- save granulo/mono-cytes --------
        file_name = "Step3_gm_" + self.export_fname
        dclab.set_temporary_feature(rtdc_ds=ds_gm,
                                    feature="userdef0",
                                    data=gm_indexes)
        ds_gm.export.hdf5(path=self.output_folder / file_name,
                          filtered=True,
                          features=self.new_features,
                          override=True)
        print("Granulocytes and Monocytes after step3 saved as", file_name)
        N_gm = len(gm_indexes)
        self.df_count.loc["Count", "Granulo-Monocytes"] = N_gm

        # Lymphocyte cluster in wbc
        lym_label = 1 - gm_label
        subset_indexes_to_select = (df["Clusters"] == lym_label).values
        ds_lym = filter_manual(ds_wbc, subset_indexes_to_select)
        lym_indexes = wbc_indexes[subset_indexes_to_select]

        # ----- save Lymphocytes before cleaning --------
        if save_unclean_lym:
            file_name = "Step3_lym_unclean_" + self.export_fname
            dclab.set_temporary_feature(rtdc_ds=ds_lym,
                                        feature="userdef0",
                                        data=lym_indexes)
            ds_lym.export.hdf5(path=self.output_folder / file_name,
                               filtered=True,
                               features=self.new_features,
                               override=True)
            print("Lymphocytes after step 3 saved as", file_name)

        # Clean Lymphocytes from micro-clots and stiff RBC
        boolian_indexes, ds_lym = filter_rtdc(ds_lym,
                                              filter_names=["tex_con_avg"],
                                              min_vals=[0],
                                              max_vals=[250])
        lym_indexes = lym_indexes[boolian_indexes]

        relevat_feat = ["bright_bc_avg", "tex_con_avg"]
        df = gmm_clustering(ds_lym, relevat_feat, 2)

        # selecting Lym cluster
        right_cluster = (df["bright_bc_avg"] > np.percentile(
            df["bright_bc_avg"], 10)).values
        lym_cluster = int(df["Clusters"][right_cluster].median())

        # Plot figure
        plt.figure()
        sns.scatterplot(data=df, x=relevat_feat[0], y=relevat_feat[1],
                        hue="Clusters")
        plt.title(
            "Lymphocytes cleaning form rest of stiff RBC & micro-clots \n Lymphocytes cluster: " + str(lym_cluster))
        plt.savefig(self.output_folder / "figures" / "Step4 lymphocytes cleaning.jpg",
                    dpi=250)

        subset_indexes_to_select = (df["Clusters"] == lym_cluster).values
        ds_clean_lym = filter_manual(ds_lym, subset_indexes_to_select)
        clean_lym_indexes = lym_indexes[subset_indexes_to_select]
        print("Step4: cleaning Lymphocytes is done.")

        # ----- save clean Lymphocytes --------
        if len(ds_clean_lym["area_um"]) > 1:
            file_name = "lym_" + self.export_fname
            dclab.set_temporary_feature(rtdc_ds=ds_clean_lym,
                                        feature="userdef0",
                                        data=clean_lym_indexes)
            ds_clean_lym.export.hdf5(path=self.output_folder / file_name,
                                     filtered=True,
                                     features=self.new_features,
                                     override=True)
            print("Lymphocytes file saved as", file_name)
            N_lym = len(clean_lym_indexes)

            self.df_count.loc["Count", "Lymphocytes"] = N_lym
            Lym_br = ds_clean_lym["bright_bc_avg"][:].mean()
            self.df_count.loc["Cell avg brightness", "Lymphocytes"] = Lym_br

            # ----- plot kde area-deform for Lym
            plt.figure()
            plot_defor_area(ds_wbc, cmap_col="viridis", plot_num=False)
            plot_defor_area(ds_clean_lym, cmap_col="spring",
                            plot_num=True)
            plt.title("Lymphocytes")
            plt.savefig(self.output_folder / "figures" / "Lymphocytes.jpg",
                        dpi=250)

        plt.close('all')
        ds_wbc.close()
        ds_gm.close()
        ds_lym.close()
        ds_clean_lym.close()

    def cluster_gm(self, gmm_covar="full"):
        print("Granulocytes and Monocytes clustering started")
        if self.perceed:
            # read GM file
            file_name = "Step3_gm_" + self.export_fname
            ds_gm = dclab.new_dataset(self.output_folder / file_name)
            gm_indexes = ds_gm["userdef0"][:]
            self.new_features = list(ds_gm.features_innate)
            wbc_file_name = "wbc_" + self.export_fname
            ds_wbc = dclab.new_dataset(self.output_folder / wbc_file_name)
        else:
            return
        if len(gm_indexes) < 100:
            self.perceed = False
            return

        # ------ break Granulomonocytes in brightness groups ----------------
        relevat_feat = ["bright_avg",
                        "tex_den_avg",
                        "tex_sva_avg"
                        ]
        df = gmm_clustering(ds_gm, relevat_feat, 3, cov_type=gmm_covar)
        df["area_um_raw"] = ds_gm["area_um_raw"][:]

        # Neutrphil cluster is the most numerous cluster
        n_0 = (df["Clusters"] == 0).sum()
        n_1 = (df["Clusters"] == 1).sum()
        n_2 = (df["Clusters"] == 2).sum()
        neut_cluster = np.argmax([n_0, n_1, n_2])

        # find clusters based on cut-off
        Neutro_min = df.loc[df["Clusters"] == neut_cluster,
                            "tex_den_avg"].min()
        Neutro_max = df.loc[df["Clusters"] == neut_cluster,
                            "tex_den_avg"].max()

        df.loc[df["tex_den_avg"] < Neutro_min, "GM group"] = "Mono"
        df.loc[df["tex_den_avg"] > Neutro_min, "GM group"] = "Neutro"
        df.loc[df["tex_den_avg"] > Neutro_max, "GM group"] = "Eos"

        plt.figure()
        sns.scatterplot(data=df, x="area_um_raw", y="tex_den_avg",
                        hue="GM group")
        plt.title("Granulocytes and Monocytes groups")
        file_name = "Step6 cLustering granulocytes&monocytes.jpg"
        plt.savefig(self.output_folder / "figures" / file_name,
                    dpi=250)

        # ---------- save Neutrophils -----------
        gran_indexes = ds_gm["userdef0"]
        subset_indexes_to_select = (df["GM group"] == "Neutro").values
        gran_indexes = gm_indexes[subset_indexes_to_select]
        ds_neu = filter_manual(ds_gm, subset_indexes_to_select)
        file_name = "neu_" + self.export_fname
        ds_neu.export.hdf5(path=self.output_folder / file_name,
                           filtered=True,
                           features=self.new_features,
                           override=True)
        print("Neutrophils saved as", file_name)
        N_neu = len(gran_indexes)
        column_name = "Neutrophils"
        self.df_count.loc["Count", column_name] = N_neu

        Neu_br = ds_neu["bright_bc_avg"][:].mean()

        # plot kde area-deform for Granulocytes
        plt.figure()
        plot_defor_area(ds_wbc, cmap_col="viridis", plot_num=False)
        plot_defor_area(ds_neu, cmap_col="autumn", plot_num=True)
        plt.title("Neutrophils")
        file_name = "Neutrophils.jpg"
        plt.savefig(self.output_folder / "figures" / file_name,
                    dpi=250)

        plt.figure()
        plot_time_and_feature(ds_neu, "deform")
        plot_line_for_time(ds_neu, "deform")
        plt.title("Neutrophil defromation over time")
        file_name = "Neutrophils deformation.jpg"
        plt.savefig(self.output_folder / "figures" / file_name,
                    dpi=250)

        # ----------- save monocytes ------
        file_name = "Step3_gm_" + self.export_fname
        ds_gm = dclab.new_dataset(self.output_folder / file_name)
        gm_indexes = ds_gm["userdef0"]
        subset_indexes_to_select = (df["GM group"] == "Mono").values
        if subset_indexes_to_select.sum() > 0:
            mono_indexes = gm_indexes[subset_indexes_to_select]
            ds_mono = filter_manual(ds_gm, subset_indexes_to_select)
            file_name = "mono_" + self.export_fname
            ds_mono.export.hdf5(path=self.output_folder / file_name,
                                filtered=True,
                                features=self.new_features,
                                override=True)
            print("Monocytes saved as", file_name)
            N_mono = len(mono_indexes)
            column_name = "Monocytes"
            self.df_count.loc["Count", column_name] = N_mono
            Mono_br = ds_mono["bright_bc_avg"][:].mean()

            # plot kde area-deform for Mono
            if subset_indexes_to_select.sum() > 6:
                plt.figure()
                plot_defor_area(ds_wbc, cmap_col="viridis",
                                plot_num=False)
                plot_defor_area(ds_mono, cmap_col="Wistia",
                                plot_num=True)
                plt.title("Monocytes")
                file_name = "Monocytes.jpg"
                plt.savefig(self.output_folder / "figures" / file_name,
                            dpi=250)
            # close the hdf5 file
            ds_mono.close()

        else:
            N_mono = 0
            Mono_br = 0

        # ------ save Eosinophils -----------
        file_name = "Step3_gm_" + self.export_fname
        ds_gm = dclab.new_dataset(self.output_folder / file_name)
        gm_indexes = ds_gm["userdef0"]
        subset_indexes_to_select = (df["GM group"] == "Eos").values
        if subset_indexes_to_select.sum() > 0:
            eos_indexes = gm_indexes[subset_indexes_to_select]
            ds_eos = filter_manual(ds_gm, subset_indexes_to_select)
            file_name = "eos_" + self.export_fname
            ds_eos.export.hdf5(path=self.output_folder / file_name,
                               filtered=True,
                               features=self.new_features,
                               override=True)
            print("Eosinophils saved as", file_name)
            N_eos = len(eos_indexes)
            column_name = "Eosinophils"
            self.df_count.loc["Count", column_name] = N_eos

            Eos_br = ds_eos["bright_bc_avg"][:].mean()

            # ------- plot kde area-deform for Eosinophils
            if subset_indexes_to_select.sum() > 6:
                plt.figure()
                plot_defor_area(ds_wbc, cmap_col="viridis",
                                plot_num=False)
                plot_defor_area(ds_eos, cmap_col="autumn",
                                plot_num=True)
                plt.title("Eosinophils")
                file_name = "Eosinophils.jpg"
                plt.savefig(self.output_folder / "figures" / file_name,
                            dpi=250)
            # close the hdf5 file
            ds_eos.close()
        else:
            N_eos = 0
            Eos_br = 0

        # ----------- save percent of wbc --------------------
        N_lym = self.df_count.loc["Count", "Lymphocytes"]
        N = N_lym + N_neu + N_mono + N_eos
        self.df_count.loc["Count", "Leukocytes"] = N
        self.df_count.loc["Percent", "Lymphocytes"] = (N_lym/N)*100

        column_name = "Neutrophils"
        self.df_count.loc["Percent", column_name] = (N_neu/N)*100
        self.df_count.loc["Cell avg brightness", column_name] = Neu_br

        column_name = "Monocytes"
        self.df_count.loc["Percent", column_name] = (N_mono/N)*100
        self.df_count.loc["Cell avg brightness", column_name] = Mono_br

        column_name = "Eosinophils"
        self.df_count.loc["Percent", column_name] = (N_eos/N)*100
        self.df_count.loc["Cell avg brightness", column_name] = Eos_br

        file_name = "count_wbc_" + self.export_fname.split(".")[0] + ".xlsx"
        self.df_count.to_excel(self.output_folder / file_name)

        plt.close('all')
        ds_wbc.close()
        ds_gm.close()
        ds_neu.close()

    def all_cell_types(self):
        self.filter_wbc()
        self.cluster_wbc()
        self.cluster_gm(gmm_covar="full")


def read_main_rtdc_file(rtdc_file_name, scalar_feat):
    print("reading file:", rtdc_file_name)
    good_rtdc = True
    # read the rtdc file
    try:
        ds = dclab.new_dataset(rtdc_file_name)
    except FileNotFoundError:
        print("Missing the file or file not readable.")
        good_rtdc = False
        ds, new_features = None, None
        return ds, new_features, good_rtdc
    # check if measurement contains enough events
    if len(ds["area_um"]) < 10**3:
        print("File is too small and it will not be processed")
        good_rtdc = False
        ds, new_features = None, None
        return ds, new_features, good_rtdc

    new_features = list(ds.features_innate)
    new_features.append("userdef0")

    if scalar_feat:
        if "image" in new_features:
            new_features.remove("image")
            new_features.remove("mask")
        else:
            print("Original file does not contain images")
        if "image_bg" in new_features:
            new_features.remove("image_bg")

    return ds, new_features, good_rtdc
