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
        default=20,
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
    parser.add_argument(
        "-e",
        "--env_shuff",
        dest="env_shuff",
        help="shuffle trials across environments",
        action="store_true",
    )
    parser.set_defaults(silent=False)
    return parser.parse_args()


def split_trace_by_env(traces, x, odors, env_mapping):
    logging.info("Finding responsive glomeruli for each odor...")
    resp_dict = imaging.resp_glom(traces, env_mapping, x, odors)
    not_blank = odors != "blank"
    odor_keep = odors[not_blank]
    trace_env = {}
    for mouse, trace_mat in traces.items():
        # glomeruli that responded to at least 2 odors
        resp = resp_dict[mouse]["sig_keep"][not_blank].sum(0) > 1
        this_trace = imaging.odor_mean(trace_mat, x)[not_blank][:, :, resp]
        v = env_mapping[mouse]
        trace_env[mouse] = {e: this_trace[:, vv] for e, vv in v.items()}
    return odor_keep, trace_env


def make_x_y(env_mapping, odor_keep, trace_env, N_RESTART, ENVS):
    logging.info("Preparing train-test split pseudopopulations...")
    HOME, ENV = ENVS
    N_ODORS = len(odor_keep)
    xon = np.arange(N_ODORS)[:, None]
    # number of trials to subselect for training and testing for each environment
    n_trials_env = {ENV: (24, 6), HOME: (12, 6)}
    # number of times to repeat each training and test trial
    # makes 24 training trials and 60 test trials in total for each env
    n_rep = {ENV: (1, 10), HOME: (2, 10)}
    trial_dict = {}
    n_tot = {}
    # subset of train/test trials from n_trials for each mouse/condition
    # to make the same number pseudo-trial pseudopopulations for each env
    for mouse, v in env_mapping.items():
        trial_dict[mouse] = {}
        for env, vv in v.items():
            tmp = np.random.rand(vv.sum(), N_RESTART).argsort(0)
            tr, te = n_trials_env[env]
            nr, ne = n_rep[env]
            n_tot[env] = (tr * nr, te * ne)
            trial_dict[mouse][env] = {
                "tr": np.repeat(tmp[:tr], nr, axis=0),
                "te": np.repeat(tmp[-te:], ne, axis=0),
            }
    all_data = []
    for i_restart in range(N_RESTART):
        tr_test_dict = {}
        for _env in ENVS:
            nr, ne = n_tot[_env]
            # get train and test_trials
            trains = []
            tests = []
            for _mouse, these_traces in trace_env.items():
                this_restart = trial_dict[_mouse][_env]
                env_traces = these_traces[_env]
                train_traces = env_traces[:, this_restart["tr"][:, i_restart]]
                # permute each odor-glom across trials
                # break correlation across glomerulation from the same trial
                # so can combine results across mice
                perms = np.random.rand(N_ODORS, nr).argsort(1)
                train_perm = train_traces[xon, perms]
                trains.append(train_perm)
                # repeat for test
                test_traces = env_traces[:, this_restart["te"][:, i_restart]]
                perms = np.random.rand(N_ODORS, ne).argsort(1)
                test_perm = test_traces[xon, perms]
                tests.append(test_perm)
            X_train = np.concatenate(trains, axis=-1)
            X_test = np.concatenate(tests, axis=-1)
            n_gloms = X_train.shape[-1]
            # flatten odors x trial axex to a single dimension
            # to make a 2D matrix of obs x glom for classification
            X_train_flat = X_train.reshape(-1, n_gloms)
            X_test_flat = X_test.reshape(-1, n_gloms)
            y_train = np.repeat(xon, nr)
            y_test = np.repeat(xon, ne)
            tr_test_dict[_env] = {
                "X": (X_train_flat, X_test_flat),
                "y": (y_train, y_test),
            }
        all_data.append(tr_test_dict)
    return all_data


def make_results_df(results, ENVS, gloms_to_subset):
    """Make classification results into dataframe"""
    res_stack = np.concatenate(results, axis=2)
    # each train/test env pair
    names = []
    for a, b in itertools.product(ENVS, ENVS):
        names.append(f"{a}: {b}")
    ids = ["n_glom", "kind", "iter"]
    dfs = {}
    for l, n in enumerate(gloms_to_subset):
        tmp = {
            to_shuff: pd.DataFrame(res_stack[s, :, :, l].T)
            for s, to_shuff in enumerate(("observed", "shuffled"))
        }
        dfs[n] = pd.concat(tmp)
    df_all = pd.concat(dfs).reset_index()
    df_all.columns = ids + names
    df_melt = df_all.melt(id_vars=ids)
    df_melt["v100"] = df_melt["value"] * 100
    return df_melt


def main(N_BOOT, N_RESTART, MAX_GLOM, out_fn, ENV_SHUFF, ENVS=("home-cage", "envA")):
    env_mapping, odors, x, traces = clf.load_data()
    odor_keep, trace_env = split_trace_by_env(traces, x, odors, env_mapping)
    all_data = make_x_y(env_mapping, odor_keep, trace_env, N_RESTART, ENVS)
    gloms_to_subset = np.arange(start=5, stop=MAX_GLOM + 1, step=5)
    clf_func = partial(
        clf.run_odor_clf,
        n_boot=N_BOOT,
        gloms_to_subset=gloms_to_subset,
        env_shuff=ENV_SHUFF,
    )
    n_cores = get_cores()
    logging.info(
        f"Running environment decoding ({len(all_data)} tasks) in parallel with {n_cores} cores..."
    )
    results = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(clf_func)(dat) for dat in all_data
    )
    df_melt = make_results_df(results, ENVS, gloms_to_subset)
    if out_fn is None:
        data_fold = get_data_folders()
        if ENV_SHUFF:
            out_fn = data_fold.results / "Odor_decoding_results_shuffled_env.csv"
        else:
            out_fn = data_fold.results / "Odor_decoding_results_no_shuffle.csv"
    df_melt.to_csv(out_fn)
    logging.info(f"Saved results to {out_fn}.")


if __name__ == "__main__":
    args = parse_args()
    # unpack arguments from argparse and then pass into main function
    main(args.n_boot, args.n_restart, args.max_glom, args.out_fn, args.env_shuff)