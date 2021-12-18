from collections import namedtuple
import os
from pathlib import Path
import re


def get_cores():
    try:
        cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        print(f"Using {cores} cores from SLURM job.")
    except KeyError as ke:
        cores = os.cpu_count()
        print(f"Not on SLURM, setting cores to max cpus ({cores})")
    return cores


def get_data_path():
    # data folder in base directory
    return Path(__file__).parents[2] / "data"


def get_data_folders(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    fields = ["processed", "raw", "tables", "results"]
    dir = namedtuple("dir", fields)
    data_fold = dir(*[data_path / f for f in fields])
    return data_fold


def find_raw_count_files(data_fold, geo="GSE173947", count_base="umi_counts.csv.gz"):
    """Look for count files in raw folder with associated metadata files"""
    raw_dict = {}
    for f in data_fold.raw.rglob(f"*{count_base}"):
        # extract experiment name
        name = re.split(f"{geo}_(.*)_{count_base}*", f.name)[1]
        raw_dict[name] = f
    # make sure we also have corresponding metadata file for each
    files = {}
    for name, count_file in raw_dict.items():
        meta_file = data_fold.raw / f"{geo}_{name}_metadata.csv.gz"
        if meta_file.exists():
            files[name] = (count_file, meta_file)
        else:
            print(f"Didn't find {meta_file} for {name}, skipping.")
    return files