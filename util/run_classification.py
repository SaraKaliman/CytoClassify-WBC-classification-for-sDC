# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:07:21 2024

@author: skalima
"""

from pathlib import Path
from classifier.wbc_classifier import wbc_classifier
from basins.populate_basin_features import process_file
import gc
import h5py


def run_classification(rtdc_files_folder, file_name,
                       output_folder,
                       export_fname,
                       out_scalar_feat,
                       add_basin_feat):
    # Run classifier
    print("Classifier will be run")
    the_classifier = wbc_classifier(rtdc_files_folder, file_name,
                                    output_folder,
                                    export_fname,
                                    out_scalar_feat,
                                    add_basin_feat)
    the_classifier.all_cell_types()

    file_path = rtdc_files_folder / file_name
    f = open(output_folder / "org_file_path.txt", "w")
    f.write(str(file_path))
    f.close()

    if add_basin_feat:
        # close all rtdc (hdf5) files after filtering
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, h5py.File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass

        path_in = Path(output_folder)
        processed = 0
        inputted = 0
        for pp in path_in.rglob("*.rtdc"):
            inputted += 1
            if process_file(pp):
                processed += 1
        print(f"Processed {processed} of {inputted} files")