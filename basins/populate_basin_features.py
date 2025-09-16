"""Write mapped basin features to .rtdc files in-place

Based on a directory structure that we use in the DC Supergroup,
this script writes mapped basin features to an existing .rtdc file.
Version 2024.04.24 added writing of S3 basins if the original .dvc
file is present.

Structure:

 - a directory contains an .rtdc file
 - this .rtdc file has the "userdef0" feature defined which
   contains the indices of the data in the file from the
   perspective of some original data file from which the
   data was exported.
 - the directory contains the "org_file_path.txt" file which
   contains a path to the original file
"""
import pathlib
import time


import dclab
import h5py


__version__ = "2024.04.24"


def get_s3_urls(local_path):
    """Assemble S3 URLs for a path that is under DVC version control"""
    local_path = pathlib.Path(local_path)
    path_dvc = local_path.with_name(local_path.name + ".dvc")
    md5 = None
    if path_dvc.exists():
        # Retrieve the MD5 sum of the original file from the .dvc file
        md5 = get_md5sum_from_dvc(path_dvc)
    # TODO: We could optionally also md5sum the original file right here.
    urls = []
    if md5:
        endpoint = "https://objectstore.hpccloud.mpcdf.mpg.de"
        urls += [
            f"{endpoint}/mpl-guck-rtdc-dvc/files/md5/{md5[:2]}/{md5[2:]}",
            f"{endpoint}/mpl-guck-rtdc-dvc/{md5[:2]}/{md5[2:]}"
            ]
    return urls


def get_md5sum_from_dvc(path_dvc):
    """Extract the MD5 checksum from a .dvc file"""
    for line in path_dvc.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line.count("md5:"):
            md5 = line.split(":")[1].strip()
            break
    else:
        print(f"Could not extract MD5 from {path_dvc}")
        md5 = None
    return md5


def process_file(path):
    """Write a mapped basin to the file"""
    path_orig = path.with_name("org_file_path.txt")

    # perform some checks
    if not path_orig.exists():
        print(f"Ignoring {path}: No original file path")
        return
    try:
        with h5py.File(path) as h5:
            if "logs" in h5 and "wrote-mapped-basin-with-s3" in h5["logs"]:
                print(f"Ignoring {path}: Already processed")
                return
            if "userdef0" not in h5["events"]:
                print(f"Ignoring {path}: No userdef0 feature")
                return
    except BaseException as e:
        print(f"Ignoring {path}: Error opening file: {e}")
        return

    # Store the mapped basin in the file
    with dclab.RTDCWriter(path, mode="replace") as hw:
        basinmap = hw.h5file["events/userdef0"][:]
        location = path_orig.read_text().strip()

        # Store file-based basin (e.g. data on drive P:\\)
        hw.store_basin(
            basin_name="Original dataset on disk",
            basin_type="file",
            basin_format="hdf5",
            basin_locs=[location],
            basin_descr="Basin added after GMMC-analysis using "
                        + "Paul's script",
            basin_map=basinmap,
            verify=False,
            )

        # Store S3-based basin;
        # Here we have a few prerequisites:
        # - The `location` must be accessible
        # - A `location.dvc` file must exist containing
        #   the MD5 sum of the original file
        # - The original input file must be present in the
        #   dedicated S3 object store `mpl-guck-rtdc-dvc`
        s3_urls = get_s3_urls(location)
        if s3_urls:
            hw.store_basin(
                basin_name="Original dataset on S3",
                basin_type="remote",
                basin_format="s3",
                basin_locs=s3_urls,
                basin_descr="Basin added after GMMC-analysis using "
                            + "Paul's script",
                basin_map=basinmap,
                verify=False,
                )

        log_name = "wrote-mapped-basin" + ("-with-s3" if s3_urls else "")

        hw.store_log(log_name,
                     [f"date: {time.ctime()}",
                      f"dclab: {dclab.__version__}",
                      f"script: {__version__}"]
                     )

    return 1


if __name__ == "__main__":
    path_in = pathlib.Path("D:/dc_analysis/LongCovid/BH/GMM_v5")
    if not path_in.exists():
        raise ValueError(f"Specified location {path_in} does not exist")

    processed = 0
    inputted = 0

    print(f"Basin complementation version {__version__}")

    if path_in.is_file():
        inputted += 1
        if process_file(path_in):
            processed += 1
    else:
        for pp in path_in.rglob("*.rtdc"):
            inputted += 1
            if process_file(pp):
                processed += 1

    print(f"Processed {processed} of {inputted} files")

