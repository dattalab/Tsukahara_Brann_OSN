import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from osn.olfr.olfr import load_func_OR


def get_bad_ribo_genes(var_names, bad=True):
    """Batch effect genes to exclude during normalization."""
    exclude = var_names.str.contains("^Rpl|^Rps")
    bad_ribo_genes = [
        "Gm42418",
        "CT010467.1",
        "mt-Rnr1",
        "mt-Rnr2",
        "Gm10076",
        "Mir6236",
        "Lars2",
        "Hint1",
        "Tpt1",
        "Fau",
        "Malat1",
        "Gm37376",
        "Gm20417",
        "Chchd2",
        "Msi2",
        "Hsp90ab1",
        "Hsp90aa1",
    ]
    if bad:
        exclude = (exclude) | (var_names.isin(bad_ribo_genes))
    return exclude


def high_expr_genes(
    var_names,
    top_genes=["Malat1", "mt-Rnr1", "mt-Rnr2", "mt-Cytb", "Gm42418", "Tmsb4x", "Xist"],
):
    return var_names.isin(top_genes)


def mito_genes(var_names):
    return var_names.str.contains("^mt-")


def exclude_norm_genes(var_names, ribo=True, high=True, mito=True):
    exclude = np.zeros_like(var_names, dtype=bool)
    if ribo:
        exclude = (exclude) | get_bad_ribo_genes(var_names, bad=False)
    if high:
        exclude = (exclude) | high_expr_genes(var_names)
    if mito:
        exclude = (exclude) | (mito_genes(var_names))
    return exclude


def get_OR_mt_gene(var_names, high=True, ribo=True):
    func_OR = load_func_OR()
    exclude = var_names.isin(func_OR.index) | (mito_genes(var_names))
    if high:
        exclude = (exclude) | high_expr_genes(var_names)
    if ribo:
        exclude = (exclude) | (get_bad_ribo_genes(var_names))
    return exclude


def norm_total(adata, target_sum=1e4, make_csr=True, **kwargs):
    """Perform total-counts (TPT) normalization
    Divide sum of UMIs (excluding genes like mt-Rnr1) by `target_sum`
    The resulting value of 1 = 1 UMI per `target_sum`"""
    if np.median(adata.X.sum(1).A) == target_sum:
        print("Already TPT normalized, using raw counts")
        adata.X = adata.raw.X.copy()
    elif isinstance(adata.X, sparse.spmatrix) and np.mod(adata.X.data, 1).any():
        raise ValueError(
            "Don't know how to normalize non-integer counts for sparse matrices"
        )
    elif isinstance(adata.X, np.ndarray) and np.mod(adata.X[adata.X != 0], 1).any():
        raise ValueError(
            "Don't know how to normalize non-integer counts for numpy arrays"
        )
    bad_genes = exclude_norm_genes(adata.var_names, **kwargs)
    tot = adata.X[:, ~bad_genes].sum(1)
    if isinstance(tot, np.matrix):
        tot = tot.A
    adata.obs["good_total_counts"] = tot
    print(f"Normalizing {type(adata.X)} count matrix")
    if isinstance(adata.X, np.ndarray):
        adata.X = (adata.X / tot[:, np.newaxis]) * target_sum
    else:
        adata.X = adata.X.multiply(1 / tot) * target_sum
        if make_csr:
            adata.X = sparse.csr_matrix(adata.X)
    return adata


def qc(adata):
    """Quality control for AnnData object. wrapper for `sc.pp.calculate_qc_metrics`"""
    adata.var = pd.DataFrame(index=adata.var_names)
    adata.var["mito"] = mito_genes(adata.var_names)
    adata.var["ribo"] = adata.var_names.str.contains("^Rp[ls]")
    adata.var["high_expr"] = high_expr_genes(adata.var_names)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mito", "ribo", "high_expr"],
        percent_top=[10, 50, 100],
        inplace=True,
    )