import argparse
from functools import partial
import itertools
import logging
from pathlib import Path
import sys

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from osn.bulb import imaging, clf
from osn.preprocess import get_data_folders, get_cores

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def parse_args():
    # parse args
    parser = argparse.ArgumentParser()
    path_type = lambda p: Path(p).absolute()
    parser.add_argument(
        "-n",
        "--n_boot",
        dest="n_boot",
        help="number of times to randomly pick gloms",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-r",
        "--n_restart",
        dest="n_restart",
        help="number of train/test trial splits",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-m",
        "--max_glom",
        dest="max_glom",
        help="maximum glom to subselect",
        type=int,
        default=50,
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


def load():
    logging.info("Loading traces...")
    _, env_mapping = imaging.load_df_meta()
    ODORS = imaging.load_key(key="odors")
    x = imaging.load_key("data/x")
    traces = imaging.load_traces()
    traces = imaging.baseline_stack(traces, x)
    return env_mapping, ODORS, x, traces


def split_trace_by_odor_env(traces, x, odors, env_mapping):
    logging.info("Finding responsive glomeruli for each odor...")
    resp_dict = imaging.resp_glom(traces, env_mapping, x, odors)
    not_blank = odors != "blank"
    odor_keep = odors[not_blank]
    # subset traces by env
    trace_env = {}
    for mouse, trace_mat in traces.items():
        v = env_mapping[mouse]
        this_trace = imaging.odor_mean(trace_mat, x)[not_blank]
        trace_env[mouse] = {e: this_trace[:, vv] for e, vv in v.items()}
    # subset traces by odor
    odor_mean_list = []
    for odor, _ in enumerate(odor_keep):
        out_dict = {}
        for mouse, v in trace_env.items():
            is_sig = resp_dict[mouse]["sig_keep"][not_blank][odor]
            out_dict[mouse] = {k: vv[odor][:, is_sig] for k, vv in v.items()}
        odor_mean_list.append(out_dict)
    return odor_keep, odor_mean_list


def make_x_y(env_mapping, odor_mean_list, CONSTANTS):
    logging.info("Preparing train-test split pseudopopulations...")
    N_BOOT, N_RESTART, MAX_GLOM, TRAIN, TEST = CONSTANTS
    n_train = TRAIN * 2
    n_test = TEST * 2
    y_train = np.repeat([0, 1], TRAIN)
    y_test = np.repeat([0, 1], TEST)
    ys = (y_train, y_test)
    gloms_to_subset = np.concatenate(
        (
            np.arange(start=1, stop=10),
            np.arange(10, MAX_GLOM + 1, step=5),
        ),
        axis=None,
    )
    trial_dict = {}
    # subset of train/test trials from n_trials for each mouse/condition
    for mouse, v in env_mapping.items():
        trial_dict[mouse] = {}
        for env, vv in v.items():
            tmp = np.random.rand(vv.sum(), N_RESTART).argsort(0)
            trial_dict[mouse][env] = {"tr": tmp[:TRAIN], "te": tmp[-TEST:]}
    # get training and test pseudopopulations
    train_test_tups = []
    for this_odor in odor_mean_list:
        trains = []
        for mouse, v in this_odor.items():
            tmp = []
            for env, vv in v.items():
                tr_idx = trial_dict[mouse][env]["tr"]
                obs = vv[tr_idx]
                # permute each glom across TRAIN trials
                perms = np.random.rand(TRAIN, N_RESTART).argsort(0)
                tmp.append(obs[perms, np.arange(N_RESTART)])
            trains.append(np.concatenate(tmp))
        tests = []
        for mouse, v in this_odor.items():
            tmp = []
            for env, vv in v.items():
                te_idx = trial_dict[mouse][env]["te"]
                obs = vv[te_idx]
                # permute each glom across TEST trials
                perms = np.random.rand(TEST, N_RESTART).argsort(0)
                tmp.append(obs[perms, np.arange(N_RESTART)])
            tests.append(np.concatenate(tmp))
        X_train_all = np.moveaxis(np.concatenate(trains, axis=-1), 1, -1)
        X_test_all = np.moveaxis(np.concatenate(tests, axis=-1), 1, -1)
        n_glom_odor = X_train_all.shape[1]
        rands = np.random.rand(n_glom_odor, N_BOOT).argsort(0)
        X_train_rand = X_train_all[:, rands].reshape(n_train, n_glom_odor, -1)[
            :, :MAX_GLOM
        ]
        X_test_rand = X_test_all[:, rands].reshape(n_test, n_glom_odor, -1)[
            :, :MAX_GLOM
        ]
        train_test_tups.append((X_train_rand, X_test_rand))

    return ys, gloms_to_subset, train_test_tups


def main(N_BOOT, N_RESTART, MAX_GLOM, out_fn, TRAIN=12, TEST=6):
    CONSTANTS = (N_BOOT, N_RESTART, MAX_GLOM, TRAIN, TEST)
    # load traces and info
    env_mapping, odors, x, traces = clf.load_data()
    odor_keep, odor_mean_list = split_trace_by_odor_env(traces, x, odors, env_mapping)
    ys, gloms_to_subset, train_test_tups = make_x_y(
        env_mapping, odor_mean_list, CONSTANTS
    )
    clf_func = partial(clf.run_env_clf, ys=ys, clf=clf.make_env_pipe(), nr=N_RESTART)

    def run_pipes(tup, train_test_tups=train_test_tups):
        odor, n, to_shuff = tup
        X_train_rand, X_test_rand = train_test_tups[odor]
        return clf_func((X_train_rand[:, :n], X_test_rand[:, :n]), shuff=to_shuff)

    # decoding for each odor, for each number of glomeruli to subsample, and for observed/shuffled data
    all_combos = list(
        itertools.product(range(len(odor_keep)), gloms_to_subset, (False, True))
    )
    n_cores = get_cores()
    logging.info(
        f"Running environment decoding ({len(all_combos)} tasks) in parallel with {n_cores} cores..."
    )
    results = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(run_pipes)(tup) for tup in all_combos
    )
    df_combos = pd.DataFrame(all_combos)
    df_combos.columns = ["odor_num", "n_glom", "shuff"]
    df_combos["odor"] = odor_keep[df_combos.odor_num]
    df_combos["value"] = [np.mean(o) * 100 for o in results]
    if out_fn is None:
        data_fold = get_data_folders()
        out_fn = data_fold.results / "Env_decoding_results.csv"
    df_combos.to_csv(out_fn)
    logging.info(f"Saved results to {out_fn}.")


if __name__ == "__main__":
    args = parse_args()
    # unpack arguments from argparse and then pass into main function
    main(args.n_boot, args.n_restart, args.max_glom, args.out_fn)