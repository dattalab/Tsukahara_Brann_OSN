import argparse
import datetime
import logging
from pathlib import Path
import sys
import time
import uuid

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.fixes import delayed

from osn.olfr import olfr, clf
from osn.preprocess import get_data_folders, get_cores

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


def run_classification(df_OR, X, best_pipe, out_folder, out_fn_base, THRESH):
    this_uuid = uuid.uuid4().hex
    out_fn = out_folder / f"{out_fn_base}{this_uuid}.csv.gz"
    or_num = LabelEncoder().fit_transform(df_OR.top_Olfr)
    transform_pipe, svc = clf.split_pipe(best_pipe)
    logging.info(f"Subselecting ORs to {THRESH} cells per OR")
    seed = np.random.randint(low=1, high=(2 ** 32) - 1)
    subset_idx = olfr.subset_source_OR_indices(or_num, THRESH, seed, n_boot=1)
    X_subset = X[subset_idx, :]
    y = or_num[subset_idx]
    y_dict = {"observed": y, "shuffled": np.random.permutation(y)}
    n_ORs = len(np.unique(y))
    # get list of OR pairs
    ind = np.vstack(np.tril_indices(n_ORs, -1)).T
    n_pairs = ind.shape[0]
    logging.info(
        f"Splitting data by fold into {THRESH} folds for {n_ORs} ORs. {n_pairs} pairs (models) in total."
    )
    skf = StratifiedKFold(n_splits=THRESH)
    data = []
    for k, this_y in y_dict.items():
        for fold, (train_index, test_index) in enumerate(skf.split(this_y, this_y)):
            y_train = this_y[train_index]
            y_test = this_y[test_index]
            data.append((k, fold, y_train, y_test, train_index, test_index))

    def transform_x(in_tup, X=X_subset, pipe=transform_pipe):
        """Run transformation pipeline to transform gene to PC/LDA space for each fold.
        Fit on all training cells (rather than for each OR pair) so all cells are in the same 20D subspace"""
        y_train, _, train_index, test_index = in_tup[-4:]
        X_train = X[train_index]
        pipe.fit(X_train, y_train)
        X_train_transform = pipe.transform(X_train)
        X_test_transform = pipe.transform(X[test_index])
        return (*in_tup[:-2], X_train_transform, X_test_transform)

    def pair_classification(in_tup, ind=ind, print_every=20_000):
        """Run classification for each pair for data in each fold (in_tup)"""
        k, fold, y_train, y_test, X_train_transform, X_test_transform = in_tup
        accs = np.zeros(print_every)
        all_acc = np.zeros(len(ind))
        st = time.time()
        # for each pair of ORs
        for i_ind, this_ind in enumerate(ind):
            # print running accuracy every `print_every` models
            rem = i_ind % print_every
            if rem == 0:
                et = (time.time() - st) / 60  # in minutes
                rt = (et * n_pairs / (i_ind + 1)) - et
                print(
                    f"Fold {fold}, key {k}, evaluated {i_ind}/{n_pairs} pairs ({i_ind/n_pairs * 100:.3f} %) in {et:.2f} minutes. ~{rt:.2f} minutes left. Last {print_every}: mean: {np.mean(accs):.4f}, std: {np.std(accs):.4f}"
                )
                sys.stdout.flush()
            # (THRESH-1) cells per OR (e.g. 18 training)
            is_train_pair = np.isin(y_train, this_ind)
            # 2 test cells (1 held-out for each OR)
            is_test_pair = np.isin(y_test, this_ind)
            # accuracy of model fit on training data applied to the held-out cells
            last_accuracy = np.mean(
                svc.fit(
                    X_train_transform[is_train_pair], y_train[is_train_pair]
                ).predict(X_test_transform[is_test_pair])
                == y_test[is_test_pair]
            )
            all_acc[i_ind] = last_accuracy
            accs[rem] = last_accuracy
        df_fold = pd.DataFrame(ind, columns=["OR1", "OR2"])
        df_fold["value"] = all_acc
        df_fold["fold"] = fold
        df_fold["shuffled"] = k == "shuffled"
        return df_fold

    n_cores = min(get_cores(), len(data)) // THRESH * THRESH
    logging.info(f"Transforming scaled data to PCA coordinates for each CV fold")
    transform_data = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(transform_x)(dat) for dat in data
    )
    logging.info(f"Running classification for each CV fold")
    dfs = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(pair_classification)(dat) for dat in transform_data
    )
    df_all = pd.concat(dfs)
    # summarize accuracy across folds
    # for each OR pair accuracy can only be in steps of 1/20 (20 cells total per pair)
    # in paper, also average across 1,000 restarts to get finer accuracy
    df_summary = df_all.groupby(["shuffled", "OR1", "OR2"], as_index=False).value.mean()
    print("Accuracy...")
    print(df_summary.groupby(["shuffled"]).value.describe())
    logging.info(f"Saving to: {out_fn}")
    df_summary.to_csv(out_fn)


def main(THRESH, NRUNS, LOW, HIGH, out_folder):
    data_fold = get_data_folders()
    now = datetime.datetime.now()
    if out_folder is None:
        out_folder = Path(
            data_fold.results, "classification", "pair", now.strftime("%Y-%m-%d")
        )
    out_fn_base = (
        f"Home_cage_pair_classification_thresh_{THRESH}_low_{LOW}_high_{HIGH}_id_"
    )
    print(f"Saving to folder: {out_folder}")
    if not out_folder.is_dir():
        out_folder.mkdir(parents=True)
    df_OR, _, X_all, _ = clf.load_and_scale_data(THRESH)
    best_pipe = clf.make_pipe(params={"f_select__low": LOW, "f_select__high": HIGH})
    i = 0
    while i < NRUNS:
        if len(list(out_folder.glob(out_fn_base + "*.csv.gz"))) > NRUNS:
            logging.info("Already found enough runs, exiting")
            break
        logging.info(f"Run {i} of {NRUNS}")
        start_time = time.time()
        try:
            run_classification(df_OR, X_all, best_pipe, out_folder, out_fn_base, THRESH)
            elapsed = time.time() - start_time
            logging.info(f"Run {i} took {elapsed:.4f} seconds")
            i += 1
        except Exception as e:
            print(e)
            elapsed = time.time() - start_time
            logging.info(f"Run {i} errored in {elapsed:.4f} seconds")
            time.sleep(5)


if __name__ == "__main__":
    # running all 20 folds in parallel requires more memory
    args = parse_args()
    # unpack arguments from argparse and then pass into main function
    main(args.thresh, args.nruns, args.low, args.high, args.out_fold)