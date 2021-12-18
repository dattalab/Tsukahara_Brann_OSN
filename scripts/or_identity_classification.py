import argparse
import datetime
import logging
from pathlib import Path
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from osn.olfr import olfr, clf
from osn.preprocess import get_data_folders

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def parse_args():
    # parse args
    path_type = lambda p: Path(p).absolute()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", dest="thresh", help="number of cells per OR", type=int, default=10
    )
    # did 1000+ restarts in the paper
    parser.add_argument(
        "-n", dest="nruns", help="number of restarts", type=int, default=100
    )
    parser.add_argument(
        "--low",
        dest="low",
        help="low percent threshold for F-score HVG selection",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--high",
        dest="high",
        help="high percent threshold for F-score HVG selection",
        type=float,
        default=10,
    )
    parser.add_argument(
        "-o",
        "--out_fold",
        dest="out_fold",
        help="output folder",
        type=path_type,
        default=None,
    )
    return parser.parse_args()


def run_classification(
    pred,
    df_OR,
    X,
    best_pipe,
    out_folder,
    out_fn_base,
    THRESH,
    percents=[0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
):
    this_uuid = uuid.uuid4().hex
    out_fn = f"{out_fn_base}{this_uuid}.pkl"
    le = LabelEncoder()
    or_labels = le.fit_transform(df_OR.top_Olfr)
    params = best_pipe.get_params()
    # subselect to same number of cells for each OR
    seed = np.random.randint(low=1, high=(2 ** 32) - 1)
    subset_idx = olfr.subset_source_OR_indices(or_labels, THRESH, seed=seed, n_boot=1)
    X = X[subset_idx, :]
    # correct OR label for each cell
    correct_labels = or_labels[subset_idx]
    y = {"observed": correct_labels, "shuffled": np.random.permutation(correct_labels)}
    pred["idx"] = subset_idx
    pred["OR"] = le.inverse_transform(correct_labels)
    pred["y"] = y
    pred["seed"] = seed
    pred["params"] = params
    i_idx, j_idx = clf.get_indices(correct_labels)
    for k, this_y in y.items():
        logging.info(f"Running classification for key: {k}")
        # sklearn pipeline
        # return svm decision boundary for each model for each cell
        # shape (8310, 344865) 831C2 for the default THRESH (10)
        dec_all = clf.run_SVM(best_pipe, X, this_y, THRESH)
        # Find all times where model A beat B
        dec_results = ((dec_all) * j_idx) + ((~dec_all) * i_idx)
        del dec_all
        # summarize results (wins for pairwise models) across ORs
        # e.g. size 8310 x 831
        pred_counts, y_pred = clf.get_predictions(dec_results, this_y)
        del dec_results
        pred[k] = {}
        # get accuracy at each percent threshold
        # 0 is perfect accuracy (out of 831 ORs)
        # 1% is how many times observed OR is within top 1% (8) of predictions
        # for multiple restarts, also average accuracy for each OSN subtype across restarts
        df_pct = (
            pd.DataFrame(clf.get_top_pct_acc(pred_counts, this_y, percents))
            .reset_index()
            .melt(id_vars="index")
        )
        print("Accuracy at each % threshold...")
        print(df_pct.groupby("variable").mean())
        df_pct["uuid"] = this_uuid
        pred[k]["pcts"] = df_pct
        pred[k]["pred_counts"] = pred_counts.astype(np.int16)
        pred[k]["top_pred"] = y_pred
        del pred_counts
    logging.info(f"Saving to: {out_fn}")
    joblib.dump(pred, out_folder / out_fn)


def main(THRESH, NRUNS, LOW, HIGH, out_folder):
    data_fold = get_data_folders()
    now = datetime.datetime.now()
    if out_folder is None:
        out_folder = Path(
            data_fold.results, "classification", "identity", now.strftime("%Y-%m-%d")
        )
    out_fn_base = (
        f"Home_cage_identity_classification_thresh_{THRESH}_low_{LOW}_high_{HIGH}_id_"
    )
    print(f"Saving to folder: {out_folder}")
    if not out_folder.is_dir():
        out_folder.mkdir(parents=True)
    df_OR, good_ORs, X_all, gene_names = clf.load_and_scale_data(THRESH)
    best_pipe = clf.make_pipe(params={"f_select__low": LOW, "f_select__high": HIGH})
    pred = {
        "olfr": good_ORs,
        "gene_names": gene_names,
        "thresh": THRESH,
        "low": LOW,
        "high": HIGH,
    }
    i = 0
    while i < NRUNS:
        if len(list(out_folder.glob(out_fn_base + "*.pkl"))) > NRUNS:
            logging.info("Already found enough runs, exiting")
            break
        logging.info(f"Run {i} of {NRUNS}")
        start_time = time.time()
        try:
            run_classification(
                pred, df_OR, X_all, best_pipe, out_folder, out_fn_base, THRESH
            )
            elapsed = time.time() - start_time
            logging.info(f"Run {i} took {elapsed:.4f} seconds")
            i += 1
        except Exception as e:
            print(e)
            elapsed = time.time() - start_time
            logging.info(f"Run {i} errored in {elapsed:.4f} seconds")
            time.sleep(5)


if __name__ == "__main__":
    # note this script generates a matrix that is shape (8310, 344865) for the default THRESH (10)
    # running all 10 folds in parallel requires >60GB memory
    args = parse_args()
    # unpack arguments from argparse and then pass into main function
    main(args.thresh, args.nruns, args.low, args.high, args.out_fold)