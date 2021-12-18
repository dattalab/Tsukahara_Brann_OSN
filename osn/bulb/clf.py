from copy import deepcopy
import itertools
import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings

from osn.bulb import imaging


def load_data():
    logging.info("Loading traces...")
    _, env_mapping = imaging.load_df_meta()
    ODORS = imaging.load_key(key="odors")
    x = imaging.load_key("data/x")
    traces = imaging.load_traces()
    traces = imaging.baseline_stack(traces, x)
    return env_mapping, ODORS, x, traces


def make_env_pipe():
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", SVC(C=0.1, kernel="linear", max_iter=20_000)),
        ]
    )
    return pipe


def make_odor_pipes():
    clf = SVC(C=0.5, kernel="linear", decision_function_shape="ovo", max_iter=20_000)
    best_pipe = Pipeline(
        [("scale", StandardScaler()), ("dim", PCA(n_components=10)), ("clf", clf)]
    )
    small_pipe = deepcopy(best_pipe)
    small_pipe.set_params(dim__n_components=5)
    return best_pipe, small_pipe


def run_env_clf(tup_in, shuff=False, ys=None, clf=None, nr=None):
    Xtr, Xte = tup_in
    n_models = Xtr.shape[-1]
    y_tr, y_te = ys
    accs = np.zeros(n_models)
    for j in range(n_models):
        if shuff:
            y_tr = np.random.permutation(y_tr)
            y_te = np.random.permutation(y_te)
        accs[j] = (clf.fit(Xtr[:, :, j], y_tr).predict(Xte[:, :, j]) == y_te).mean()
    return accs.reshape(-1, nr).mean(0)


def perm_env(x1, x3):
    nt = x1.shape[0]
    idx = (np.random.rand(nt) < 0.5).astype(int)
    x_stack = np.stack([x1, x3])
    x1 = x_stack[idx, np.arange(nt)[None, :]].squeeze()
    x3 = x_stack[1 - idx, np.arange(nt)[None, :]].squeeze()
    return x1, x3


@ignore_warnings(category=ConvergenceWarning)
def run_odor_clf(in_dict, n_boot=None, gloms_to_subset=None, env_shuff=False):
    best_pipe, small_pipe = make_odor_pipes()
    x1, x2 = in_dict["home-cage"]["X"]
    y1, y2 = in_dict["home-cage"]["y"]
    x3, x4 = in_dict["envA"]["X"]
    y3, y4 = in_dict["envA"]["y"]
    if env_shuff:
        # permute across envs
        x1, x3 = perm_env(x1, x3)
        x2, x4 = perm_env(x2, x4)
    n_glom_max = x1.shape[-1]
    # glomeruli to subsample
    subset_mat = np.random.rand(n_boot, n_glom_max).argsort(1)
    # observed/shuffled x env train/test pairs x restarts x number of glomeruli subsampled
    acc_mat = np.zeros((2, 4, n_boot, len(gloms_to_subset)))
    # for observed and shuffled odor labels
    for s, to_shuff in enumerate((False, True)):
        xp = itertools.product((x1, x3), (x2, x4))
        yp = itertools.product((y1, y3), (y2, y4))
        # every train test combo within/between envs
        for j, (_xs, _ys) in enumerate(zip(xp, yp)):
            x, xt = _xs
            y, yt = _ys
            if to_shuff:
                y = np.random.permutation(y)
                yt = np.random.permutation(yt)
            # subsample glomeruli and run for each of n_boot
            for k, subset in enumerate(subset_mat):
                xs = x[:, subset]
                xts = xt[:, subset]
                for l, n in enumerate(gloms_to_subset):
                    if n == 5:
                        pipe_use = small_pipe
                    else:
                        pipe_use = best_pipe
                    # do classification and report accuracy
                    pipe_use.fit(xs[:, :n], y)
                    acc_mat[s, j, k, l] = (pipe_use.predict(xts[:, :n]) == yt).mean()
    return acc_mat