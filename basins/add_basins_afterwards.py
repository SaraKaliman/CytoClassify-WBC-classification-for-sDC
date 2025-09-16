# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:08:50 2024

@author: skalima
"""

from pathlib import Path
from populate_basin_features import process_file
import gc
import h5py


def add_basins_afterwards(output_folder):
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


if __name__ == '__main__':
    # If you have chosen Add_basins = False and you want to add then afterwards
    output_folder = Path(r"C:\Users\<your_name>\output_folder")
    add_basins_afterwards(output_folder)
