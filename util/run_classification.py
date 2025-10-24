# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:07:21 2024

@author: skalima
"""
from classifier.wbc_classifier import wbc_classifier


def run_classification(rtdc_files_folder, file_name,
                       output_folder,
                       export_fname,
                       out_scalar_feat):
    # Run classifier
    print("Automated WBC classification will be run")
    the_classifier = wbc_classifier(rtdc_files_folder, file_name,
                                    output_folder,
                                    export_fname,
                                    out_scalar_feat)
    the_classifier.all_cell_types()

    file_path = rtdc_files_folder / file_name
    f = open(output_folder / "org_file_path.txt", "w")
    f.write(str(file_path))
    f.close()
