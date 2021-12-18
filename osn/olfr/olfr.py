import itertools

import numpy as np
import numpy_groupies as npg
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from osn.preprocess.util import get_data_folders


def load_func_OR():
    data_fold = get_data_folders()
    func_OR = pd.read_csv(data_fold.tables / "Olfr_biomart.csv").set_index(
        "external_gene_name"
    )
    return func_OR


def get_OR_info(
    adata,
    expression_thresh=3,
    diff_thresh=2,
    extra_cols=["nostril", "odor", "env", "activation_pca", "activated"],
):
    """Determine which OR is expressed in each cell.

    Args:
        adata (anndata): mature OSNs, with raw UMI counts in `adata.raw`
        expression_thresh (int, optional): number of UMIs to consider an OR expressed. Defaults to 3.
        diff_thresh (int, optional): difference between olfr with highest counts and second highest. Defaults to 2.
        extra_cols: columns in adata to add to df_OR

    Returns:
        df_OR (pd.DataFrame): rows obs_names of singly-expression OSNs and columns `top_Olfr` saying which OR is expressed. Length `has_OR.sum`()
        Columns of df_OR also include GEP usages, OR expression, and `extra_cols`.
        has_OR (np.array): boolean for which obs_names of adata have ORs. adata.obs_names[has_OR] == df_OR.index
    """
    func_OR = load_func_OR()
    is_func_or = adata.var_names.isin(func_OR.index)

    if "OR_counts" not in adata.obs or adata.obs["OR_counts"].isna().any():
        adata.obs["OR_counts"] = adata.raw.X[:, is_func_or].max(1).A.flatten()
        adata.obs["OR_counts_norm"] = (
            adata.obs["OR_counts"] / adata.obs.total_counts * 1e4
        )
    # how many cells express each OR
    above_thresh = adata.raw.X[:, is_func_or] >= expression_thresh
    n_ORs_each_cell = np.asarray((above_thresh).sum(axis=1)).flatten()
    or_sorted_counts = np.sort(adata.raw.X[:, is_func_or].A, axis=1)
    count_diff = or_sorted_counts[:, -1] - or_sorted_counts[:, -2]
    # high counts of one OR and below thresh for another
    has_OR = (n_ORs_each_cell == 1) & (count_diff >= diff_thresh)
    idx_top_Olfr = adata.raw.X[:, is_func_or].argmax(1).A[has_OR].flatten()
    top_Olfr = adata.var_names[is_func_or][idx_top_Olfr]

    extra_cols = [e for e in extra_cols if e in adata.obs]
    join_cols = ["orig_ident", "source", "OR_counts", "OR_counts_norm"] + extra_cols

    df_OR = pd.DataFrame(
        index=adata.obs_names[has_OR], data=top_Olfr, columns=["top_Olfr"]
    ).join(adata.obs[join_cols])
    df_OR = df_OR.merge(
        func_OR[["OR_class", "cluster_3Mb_bychr"]],
        left_on="top_Olfr",
        right_index=True,
        how="left",
    )
    if "X_nmf" in adata.obsm:
        df_OR = df_OR.join(adata.obsm["X_nmf"].iloc[:, :7])
        df_OR["ES_score"] = df_OR["High"] - df_OR["Low"]
    # convert back to strs since scanpy converts to categorical
    cat_cols = df_OR.columns[df_OR.dtypes == "category"]
    df_OR[cat_cols] = df_OR[cat_cols].astype(str)
    return df_OR, has_OR


def get_group_mat(df, cat="source", olfr_key="top_Olfr"):
    olfr = LabelEncoder().fit_transform(df[olfr_key])
    if cat is None:
        out = olfr
    else:
        group = LabelEncoder().fit_transform(df[cat])
        out = np.vstack((olfr, group))
    return out


def filter_OR_source(df_OR, thresh, col=None):
    """Find ORs present in at least SOURCE_THRESH cells in each of n_sources.

    Parameters
    ----------
    df_OR : pd.DataFrame from `get_OR_info`
    thresh : int
        [number of ORs for each OR for each of n_sources]
    col : str, optional
        [column of df_OR to use for grouping
    """
    uq_Olfr = np.unique(df_OR.top_Olfr)
    group_mat = get_group_mat(df_OR, cat=col)
    enough_olfr = npg.aggregate(group_mat, 1, "sum") >= thresh
    if enough_olfr.ndim > 1:
        enough_olfr = enough_olfr.all(1)
    good_ORs = uq_Olfr[enough_olfr]
    has_enough_ORs = df_OR.top_Olfr.isin(good_ORs)
    return good_ORs, has_enough_ORs


def make_combo_df(df, col="orig_ident", SOURCE_THRESH=4, combos=None, uq=None):
    """Get means for ORs found in both items in all the pairs of a given column (e.g. pairs of mice, envs).

    Args:
        df (pd.DataFrame): df_OR-like datafrae
        col (str): Column of `df_OR` to use. Defaults to "orig_ident".
        SOURCE_THRESH (int, optional): number of cells per OR for each source. Defaults to 4.
        combos Optional, specify combos. Defaults to None.
        uq Optional, specify uq items. Defaults to None.

    Returns:
        uq: unique elements in `col`
        combo_dict: dictionary with keys each pair `combos` with mean across df_OR columns for each OR found in both sources in the pair
        combos: pairs in `uq`

    """
    if uq is None:
        uq = np.unique(df[col])
    if combos is None:
        combos = list(itertools.combinations(uq, 2))
    combo_dict = {}
    for combo in combos:
        in_combo = df[col].isin(combo)
        df_combo = df[in_combo]
        _, has_enough = filter_OR_source(df_combo, SOURCE_THRESH, col=col)
        df_mean = df_combo[has_enough].groupby(["top_Olfr", col], as_index=False).mean()
        combo_dict[combo] = df_mean.set_index(["top_Olfr", col])
    return uq, combo_dict, combos


def get_boot_idx(or_num, n_boots=100, source=None):
    """Randomly chose equal numbers of cells per OR (with replacement).
    Optionally, chose equal numbers for each OR for each source.
    """
    tril_idx = np.tril_indices(n_boots, -1)
    g1 = []
    if source is None:
        for i in np.unique(or_num):
            uq_vals = np.where(or_num == i)[0]
            n_uq = len(uq_vals)
            # sample from OSNs expressing that OR with replacement
            rands = np.random.choice(n_uq, size=(n_uq, n_boots))
            g1.append(uq_vals[rands])
    else:
        uq_source = np.unique(source)
        for i in np.unique(or_num):
            is_or = or_num == i
            for s in uq_source:
                uq_vals = np.where((is_or) & (source == s))[0]
                n_uq = len(uq_vals)
                rands = np.random.choice(n_uq, size=(n_uq, n_boots))
                g1.append(uq_vals[rands])
    return np.vstack(g1), tril_idx


def subset_source_OR_indices(or_num, THRESH=4, source=None, seed=None, n_boot=100):
    """Subset equal numbers of cells from each source for each OR n_boot times

    Arguments:
        or_num {[type]} -- [np.array of or_labels to randomly sample from]

    Keyword Arguments:
        SOURCE_THRESH {int} -- [number of cells per source per OR] (default: {4})
        source {[type]} -- [vector describing source for each cell] (default: {None})
        seed {[type]} -- [random seed] (default: {None})
        n_boot {int} -- [number different indices to generate] (default: {100})
    """
    np.random.seed(seed)
    tmp = []
    for n in np.unique(or_num):
        if source is None:
            uq_vals = np.where((or_num == n))[0]
            tmp.append(
                uq_vals[np.random.rand(len(uq_vals), n_boot).argsort(0)[:THRESH, :]]
            )
        else:
            for s in np.unique(source):
                uq_vals = np.where((or_num == n) & (source == s))[0]
                tmp.append(
                    uq_vals[np.random.rand(len(uq_vals), n_boot).argsort(0)[:THRESH, :]]
                )
    # squeeze if only 1 boot
    indices = np.vstack(tmp).squeeze()
    return indices


def get_ci_vect_vectorized(x, n_boots=1000, n_samp=None, function=np.mean, pct=5):
    """Get bootstrapped confidence intervals of a vector `x`"""
    if isinstance(x, pd.core.series.Series):
        x = x.values
    pct /= 2
    n_vals = len(x)
    if n_samp is None:
        n_samp = n_vals
    boots = function(x[np.random.choice(n_vals, size=(n_samp, n_boots))], axis=0)
    return np.percentile(boots, [pct, 100 - pct])