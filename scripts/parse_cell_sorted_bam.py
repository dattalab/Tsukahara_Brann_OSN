import argparse
import gzip
import logging
import os
from pathlib import Path
import sys

import pandas as pd
import pysam

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    path_type = lambda p: Path(p).absolute()
    parser.add_argument(
        "-i",
        "--in_bam",
        dest="in_bam",
        help="input cell_sorted bam filename",
        type=path_type,
        default="cell_sorted_possorted_genome_bam.unique.bam",
    )
    parser.add_argument(
        "-o",
        "--out_fn",
        dest="out_fn",
        help="output filename",
        type=path_type,
        default=None,
    )
    return parser.parse_args()


def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def get_dge(df):
    # unique singly-mapped reads that map to single genes
    tenx = (
        (df.map_quality == "255")
        & (~df.GN.isna())
        & (~df.GN.str.contains(";", na=False))
    )
    df = df[tenx]
    genes_per_umi = df.groupby("UB").GN.nunique()
    # number of unique barcodes (UMI) per gene
    # for UMIs that only map to a single gene
    df_to_write = pd.DataFrame(
        df[df.UB.isin(genes_per_umi[genes_per_umi == 1].index)]
        .groupby("GN")
        .UB.nunique()
    )
    return df_to_write


def main(bam_fn, out_fn):
    fin = pysam.AlignmentFile(bam_fn)
    if Path(bam_fn.parent, "bcfile.tsv*").is_file():
        logging.debug(f"Found bcfile in cd")
        bcfile = Path(bam_fn.parent, "bcfile.tsv*")
    else:
        # walk down dir and find bcfile
        bcfile = None
        for dir in [
            x for x in Path(bam_fn.parent).glob("filtered_*_bc_matr*") if x.is_dir()
        ]:
            # check if bcfile in folder
            files = [x for x in dir.rglob("barcodes.tsv*") if x.is_file()]
            if len(files) > 0:
                bcfile = files[0]  # take first batch
                logging.debug(f"Found bcfile in {dir}")
                break
        if not bcfile:
            logging.debug(f"Could not find bcfile, exiting")
            sys.exit()
    # account for the case where we sometimes have -1 on the end if from 10x
    if ".gz" in bcfile.suffixes:
        valid_bcset = set(
            [s.rstrip("\n").split("-")[0] for s in gzip.open(bcfile, "rt").readlines()]
        )
    else:
        valid_bcset = set(
            [s.rstrip("\n").split("-")[0] for s in open(bcfile, "r").readlines()]
        )
    logging.debug(f"Found {len(valid_bcset)} valid barcodes")
    if out_fn is None:
        out_fn = unique_path(bam_fn.parent, "counts_no_double_umi_{:03d}.tsv.gz")
    logging.debug(bam_fn)
    logging.debug(fin)
    logging.debug(out_fn)

    reads = []
    tags = []
    current_cell = None
    # It is super slow to read the file sequentially like this. 
    # alteratively could split bam file by cell barcode and run in parallel
    # could also do reading and writing in separate threads/subprocesses
    for i, read in enumerate(fin):
        if i % 1000000 == 0:
            logging.debug(f"Read first {i // 1000000} million reads")
        if not read.has_tag("GN"):
            continue
        if read.has_tag("CB"):
            this_cell = read.get_tag("CB").split("-")[0]
            if this_cell in valid_bcset:
                this_tags = dict(read.get_tags())
                # check if it is time to summarize last cell
                # save UMI counts for each count for each cell
                if this_cell != current_cell:
                    if current_cell is not None:
                        logging.debug("Summarizing counts")
                        df = pd.concat(
                            [pd.DataFrame(reads), pd.DataFrame(tags)], axis=1
                        )
                        df_to_write = get_dge(df)
                        df_to_write["CB"] = current_cell
                        df_to_write.to_csv(
                            out_fn, sep="\t", mode="a", header=None, compression="gzip"
                        )
                        del df_to_write
                        del df
                    logging.debug(f"Found new cell {this_cell}")
                    current_cell = this_cell
                    reads = []
                    tags = []
                r = read.to_dict()
                del r["tags"]
                reads.append(r)
                tags.append(this_tags)
    # Append with remaining reads
    logging.debug("Reached end of file, summarizing last cell")
    df = pd.concat([pd.DataFrame(reads), pd.DataFrame(tags)], axis=1)
    df_to_write = get_dge(df)
    df_to_write["CB"] = this_cell
    df_to_write.to_csv(out_fn, sep="\t", mode="a", header=None, compression="gzip")
    logging.debug(f"Saved file to {out_fn}")
    logging.debug("Removing writing permissions to the file")
    logging.debug(oct(os.stat(out_fn).st_mode))
    # prevent future appending
    os.chmod(out_fn, 0o444)


if __name__ == "__main__":
    args = parse_args()
    main(args.in_bam, args.out_fn)
