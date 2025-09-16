# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:24:12 2024

@author: skalima
"""

from glob import glob
import os
from pathlib import Path
from util.run_classification import run_classification


if __name__ == '__main__':
    # --------------------- PARTS CHANGABLE BY USER -------------------------
    rtdc_files_folder = Path(r"<folder containing rtdc file(s)>")
    output_folder = Path(r"<your output folder>")

    # Chose weather you want to output cell images
    # exporting images takes time, it is faster to export scalar feat
    Output_only_scalar_features = True   # set True to not export images

    # If you export only scalar features you can add basins
    # They will point to the cell image using event index and
    # path of the original rt-dc file.
    # You need new ShapeOut2 to use basins
    Add_basins = False

    os.chdir(rtdc_files_folder)

    files = glob("*.rtdc")
    for file_name in files:
        export_fname = file_name
        new_subfol_name = file_name.split(".")[0]
        os.chdir(output_folder)
        if not os.path.isdir(new_subfol_name):
            os.mkdir(new_subfol_name)
        os.chdir(new_subfol_name)
        if not os.path.isdir("figures"):
            os.mkdir("figures")
        output_folder = output_folder / new_subfol_name
        os.chdir(rtdc_files_folder)
        file_subfolder = Path()
        run_classification(rtdc_files_folder, file_name,
                           output_folder,
                           export_fname,
                           out_scalar_feat=Output_only_scalar_features,
                           add_basin_feat=Add_basins)

