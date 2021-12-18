import argparse
from ftplib import FTP

from osn.preprocess import get_data_folders


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--silent",
        dest="silent",
        help="skip confirmation before downloading all files",
        action="store_true",
    )
    parser.set_defaults(silent=False)
    return parser.parse_args()


def main(CONFIRM):
    data_fold = get_data_folders()

    ftp = FTP("ftp.ncbi.nlm.nih.gov")
    ftp.login("anonymous", "")
    ftp.cwd("geo/series/GSE173nnn/GSE173947/suppl/")
    print("Found these supplementary files...")
    ftp.retrlines("LIST")
    files = ftp.nlst()

    # group files
    msg = "Download all files [y] or ask for confirmation for each [n]?"
    if not CONFIRM and input(msg).lower() == "n":
        CONFIRM = True

    for fn in files:
        file_path = data_fold.raw / fn
        if file_path.exists():
            print(f"Already found {fn}, skipping.")
            to_download = False
        if CONFIRM:
            to_download = input(f"Download {fn} [y]?").lower() == "y"
        else:
            to_download = True
        if to_download:
            print(f"Downloading {fn}...")
            with open(file_path, "wb") as fp:
                ftp.retrbinary(f"RETR {fn}", fp.write)


if __name__ == "__main__":
    args = parse_args()
    # unpack arguments from argparse and then pass into main function
    main(args.silent)