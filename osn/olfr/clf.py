import logging

from numba import njit, prange
import numpy as np
import numpy_groupies as npg
import pandas as pd
from tqdm import tqdm
import scanpy as sc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings

from osn.olfr import olfr
from osn.preprocess import get_data_folders
from osn.preprocess.norm import get_OR_mt_gene


def _default_clf():
    return SVC(
        decision_function_shape="ovo", gamma="scale", kernel="linear", max_iter=-1
    )


@ignore_warnings(category=ConvergenceWarning)
def run_SVM(best_pipe, X, y, folds):
    if folds >= 20:
        folds = 10
    dec_all = (
        cross_val_predict(
            best_pipe,
            X,
            y,
            cv=StratifiedKFold(n_splits=folds),
            n_jobs=folds,
            verbose=3,
            method="decision_function",
        )
        < 0
    )
    return dec_all


def _check_params(params, default_params):
    if params is None:
        params = default_params
    else:
        for k, v in default_params.items():
            if k not in params:
                params[k] = v
    return params


def make_default_params(f_select=True, lda=True):
    """Identified via grid/random search"""
    params = {"pca__n_components": 38, "svc__C": 0.03}
    if f_select:
        params["f_select__low"] = 0
        params["f_select__high"] = 10
    if lda:
        params["lda__n_components"] = 20
    return params


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Class to use for feature selection in pipeline
    Raising low threshold (from 0) removes top gene
    Lowering high threshold keeps less genes
    """
    # Class Constructor
    def __init__(self, func=f_classif, percent=True, low=0, high=50):
        self.func = func
        self.percent = percent
        self.low = low
        self.high = high

    def fit(self, X, y):
        self.F, _ = self.func(X, y)
        if self.percent:
            if not all([0 <= v <= 100 for v in (self.low, self.high)]):
                raise ValueError(
                    f"percentiles should be >=0, <=100; got {self.low} and {self.high}"
                )
            self._low = int(len(self.F) * self.low / 100)
            self._high = int(len(self.F) * self.high / 100)
        else:
            self._low = self.low
            self._high = self.high
        return self

    def transform(self, X, y=None):
        # keep low to high % of columns based on F-score selector
        return X[:, (-1 * self.F).argsort()[self._low : self._high]]


def make_pipe(params=None, scale=False):
    # use scale=True when doing classification for NMF GEPs
    # when using GEPs, PCA and LDA are no longer necessary to reduce the dimensionality
    default_params = make_default_params()
    params = _check_params(params, default_params)
    steps = [
        ("f_select", FeatureSelector()),
        ("pca", PCA(random_state=42)),
        ("lda", LDA()),
        ("svc", _default_clf()),
    ]
    if scale:
        steps = [
            ("var", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ] + steps
    pipe = Pipeline(steps)
    pipe.set_params(**params)
    return pipe


def split_pipe(best_pipe):
    """Split classification from transform part of pipe, for pairwise classification
    """
    transform_pipe = Pipeline(best_pipe.steps[:-1])
    svc = best_pipe.named_steps["svc"]
    return transform_pipe, svc


def load_gene_std(gene_names=None):
    data_fold = get_data_folders()
    df_mean_std = pd.read_csv(
        data_fold.tables / "home_cage_all_genes_mean_sd.csv.gz", index_col=0
    )
    if gene_names is None:
        return df_mean_std
    else:
        mean_std = df_mean_std.loc[gene_names].values
        means = mean_std[:, :2].T
        stds = mean_std[:, 2:].T
        return means, stds


def load_and_scale_data(THRESH):
    """Load home-cage data for classification
    Scale data using saved means and SD."""
    data_fold = get_data_folders()
    ad_fn = data_fold.processed / "home_cage_norm.h5ad"
    logging.info(f"Loading adata file and subselecting ORs in at least {THRESH} cells")
    adata = sc.read(ad_fn)
    df_OR, has_OR = olfr.get_OR_info(adata)
    good_ORs, has_enough_ORs = olfr.filter_OR_source(df_OR, THRESH)
    df_OR_keep = df_OR[has_enough_ORs]
    logging.info(f"Found {len(good_ORs)} ORs and {len(df_OR_keep)} OSNs.")
    is_cell_keep = adata.obs_names.isin(df_OR_keep.index)
    # subselect to number of variable genes
    is_OR_mt_gene = get_OR_mt_gene(adata.var_names)
    in_enough_cells = (adata.raw.X > 0).A[has_OR, :][has_enough_ORs, :].sum(
        0
    ) > has_enough_ORs.sum() * 0.005
    gene_used = (in_enough_cells) & (~is_OR_mt_gene)
    gene_names = adata.var_names[gene_used]
    label_round2 = LabelEncoder().fit_transform(adata.obs.round2)[is_cell_keep]
    means, stds = load_gene_std(gene_names)
    logging.info("Scaling genes")
    X = np.log1p(adata.X[is_cell_keep][:, gene_used].A)
    X = (X - means[label_round2]) / stds[label_round2]
    return df_OR_keep, good_ORs, X, gene_names


def get_indices(labels):
    """Return the indices to calculate top_n accuracy from an SVM.
    The SVM decision function returns distances to SVs in a format that's n_classes * (n_classes - 1) / 2.
    labels contains a list of classes that we're trying to predict.

    Returns:
        i_idx: Class A for each model
        j_ix: Class B for each model
    """
    n_classes = len(np.unique(labels))
    comp_mat = np.zeros((n_classes, n_classes), dtype=np.int16) + np.arange(
        n_classes, dtype=np.int16
    )
    tri = np.triu_indices_from(comp_mat, 1)
    i_idx = comp_mat.T[tri]
    j_idx = comp_mat[tri]
    return i_idx, j_idx


def get_predictions(dec_results, y, verbose=True):
    """Summarize dec_results (from `run_SVM`) across ORs
    Returns matrix cells x ORs with counts number of times that OR won in the models it was present
    predcounts.argmax(1) should equal y if classification is perfect, or
    when doing predcounts.argsort(1) the OR in y should be close to the top"""
    n_classes = len(np.unique(y))
    n_samples = len(y)
    assert (
        dec_results.shape[0] == n_samples
    ), "Number of predictions must match number of samples"
    pred_counts = np.zeros((n_samples, n_classes), dtype=np.int16)
    # count number of times each OR "won" in the pairs it was present
    for i in range(n_samples):
        pred_counts[i, :] = np.bincount(dec_results[i, :], minlength=n_classes)
    # get top predictor
    y_pred = pred_counts.argmax(1)
    # correct for ties
    # if OR is tied; count as accurate
    p_max = pred_counts.max(1)
    is_top = pred_counts == p_max[:, np.newaxis]
    or_is_top = is_top[(np.arange(len(y)), y)]
    y_pred[or_is_top] = y[or_is_top]
    score = metrics.balanced_accuracy_score(y, y_pred)
    if verbose:
        print(f"Accuracy was {score:.4f}")
    return pred_counts, y_pred


def get_conf_mat(ytrue, pred):
    cnf_matrix = metrics.confusion_matrix(ytrue, pred)
    cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cnf_matrix


def get_top_pct_acc(pred_counts, y, pcts):
    """Summarize accuracy of predictions for each cell in pred_counts at each pct in `pcts`"""
    n_cells = len(y)
    n_or = len(np.unique(y))
    assert (pred_counts.shape[1] == n_or) and (
        pred_counts.shape[0] == n_cells
    ), "Dimensions must agree"
    # get rank of specified label (y)
    xrange = np.arange(n_cells)
    # number of ORs with higher wins
    n_higher = (pred_counts > pred_counts[(xrange, y)][:, np.newaxis]).sum(1)
    y_pred = pred_counts.argmax(1)

    def _get_top_pct(pct):
        pred = y_pred.copy()
        if pct == 0:
            is_top_accuracy = n_higher == 0
        else:
            # correct for indexing starting at 1
            is_top_accuracy = n_higher <= (n_or * pct - 1)
        pred[is_top_accuracy] = y[is_top_accuracy]
        return np.diag(get_conf_mat(y, pred))

    return {pct: _get_top_pct(pct) for pct in pcts}


@njit(parallel=True, fastmath=True)
def jit_dist(mat, rands):
    """Distance formula for all OR subsets at once"""
    n_env = mat.shape[1]
    n_rand = rands.shape[1]
    out_mat = np.zeros((n_rand, n_env, n_env, n_rand))
    # sqrt along axis
    for i in prange(n_rand):
        out_mat[i] = np.sqrt(mat[rands[:, i]].sum(0))
    return out_mat


def es_score_decoding(df_dpg, THRESH=6, NBOOT=1000, NRAND=1000, MAX_OLFR=100):
    """Decoding of environments using ES score

    Args:
        df_dpg ([type]): df_OR. Environment DPG data
        THRESH (int, optional): 
            Number of cells per OR (in each env).
            One cell held out for testing and the rest are used for training pseudopopulations.
            Defaults to 6.
        NBOOT (int, optional): Number of restarts selecting THRESH cells per OR. Defaults to 1000.
        NRAND (int, optional): Number of restarts subselecting up to MAX_OLFR ORs to consider. Defaults to 1000.
        MAX_OLFR (int, optional): Maximum number of ORs to subset. Defaults to 100.

    Returns:
        pd.DataFrame summarizing results for each env for observed and shuffled data for each subset/restart
    """
    df_dpg = df_dpg.copy()
    df_dpg["or_env"] = df_dpg["top_Olfr"].str.cat(df_dpg.env, sep="-")
    # keep ORs in at least THRESH cells
    good_olfr, has_enough = olfr.filter_OR_source(df_dpg, THRESH, col="env")
    df_dpg_keep = df_dpg[has_enough]
    ess = df_dpg_keep.ES_score.values
    n_olfr = len(good_olfr)
    uq_env = np.unique(df_dpg_keep.env)
    n_env = len(uq_env)
    print(f"Using {n_olfr} ORs found in at least {THRESH} cells in all {n_env} envs.")
    or_labels = LabelEncoder().fit_transform(df_dpg_keep.top_Olfr)
    oe_labels = LabelEncoder().fit_transform(df_dpg_keep.or_env)
    print(f"Subsampling {THRESH} cells per OR for each of {NBOOT} restarts.")
    indices = olfr.subset_source_OR_indices(
        or_labels, THRESH, source=df_dpg_keep.env, n_boot=NBOOT
    )
    labs = oe_labels[indices[:, 0]]

    def run_min_dist(mat, index_names=["subset", "restart"], **kwargs):
        to_test = np.concatenate(
            (np.arange(start=1, stop=25), np.arange(25, MAX_OLFR + 1, step=5)),
            axis=None,
        )
        y = np.arange(n_env)[:, None]
        outs = {}
        for t in tqdm(to_test, ncols=100, **kwargs):
            # NRAND subsets of t ORs
            rands = np.random.choice(n_olfr, size=(t, NRAND))
            # minimum distance is predicted environment
            # accuracy if predicted env of held-out vector is correct
            tmp = (np.argmin(jit_dist(mat, rands), 1) == y).mean(-1) * 100
            outs[t] = pd.DataFrame(tmp, columns=uq_env)
        df_melt = pd.concat(outs, names=index_names)
        return df_melt.reset_index().melt(id_vars=index_names)

    print(
        f"Performing classification, subsampling (1–{MAX_OLFR}) ORs {NRAND} times for each restart."
    )
    # hold out one cell per OR
    is_test = np.mod(np.arange(len(labs)), THRESH) == 0
    is_train = ~is_test
    # get OR means for each OR
    # makes n_OR x n_env x NBOOT training matrix
    ess_train = npg.aggregate(
        labs[is_train], ess[indices[is_train]], func="mean", axis=0
    ).reshape(n_olfr, n_env, NBOOT)
    # n_OR x n_env x NBOOT training matrix
    ess_test = ess[indices][is_test].reshape(n_olfr, n_env, NBOOT)
    # distance between training and test vectors
    # n_olfr x n_test_env x n_train_env x n_restarts
    sqr_mat = (ess_train[:, :, None, :] - ess_test[:, None, :, :]) ** 2
    df_obs = run_min_dist(sqr_mat, desc="observed")
    # shuffled environments
    # permute the indices across the cells for each OR
    # this still has the same env across all ORs for each training/test cell
    # still keep one cell per env for testing
    n_cells_use = (THRESH - 1) * n_env
    ind_reshape = indices[is_train].reshape(n_olfr, n_cells_use, NBOOT)
    order = np.random.rand(n_cells_use, NBOOT).argsort(0)
    ind_perm = ind_reshape[:, order, np.arange(NBOOT)].reshape(-1, NBOOT)
    ess_train_shuff = npg.aggregate(
        labs[is_train], ess[ind_perm], func="mean", axis=0
    ).reshape(n_olfr, n_env, NBOOT)
    # use held-out observed test vectors with vects from shuffled data
    shuff_mat = (ess_train_shuff[:, :, None, :] - ess_test[:, None, :, :]) ** 2
    df_shuff = run_min_dist(shuff_mat, desc="shuffled")
    df_out = pd.concat({"observed": df_obs, "shuffled": df_shuff}, names=["kind"])
    return df_out.reset_index().drop(columns=["level_1"])
