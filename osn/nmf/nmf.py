import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.preprocessing import LabelEncoder

from osn.preprocess.util import get_data_folders


def load_scale_loadings():
    """Load HVG standard deviations. These were calculated using OR-balanced home-cage data.
    To apply cNMF to new datasets the TPT-normalized gene expression in the new data is divided by these home-cage SDs.
    Also loads and returns the cNMF gene loadings for each HVG (matrix of HVGs x GEPs)"""
    data_fold = get_data_folders()
    df_std = pd.read_csv(data_fold.tables / "hvg_std_for_cnmf.csv", index_col=0)
    df_loadings = pd.read_csv(
        data_fold.tables / "GSE173947_Table_S2_nmf_loadings.csv", index_col=0
    )
    return df_std, df_loadings


def load_func_genes():
    """Load annotated functional genes associated with GEP_High and GEP_Low"""
    data_fold = get_data_folders()
    df_func = pd.read_csv(
        data_fold.tables / "GSE173947_Table_S3_ES_neuronal_genes.csv", index_col=0
    )
    df_func.columns = ["gep", "cat1", "cat2", "func"]
    df_func["is_func"] = df_func.func == "YES"
    # order categories and make into categorical dtype
    cats = [
        "calcium",
        "OR signaling",
        "ion transfer",
        "protein transport",
        "axon guidance",
        "synapse",
    ]
    df_func["cat1"] = pd.Categorical(df_func.cat1, ordered=True, categories=cats)
    return df_func, cats


def make_kwargs():
    nmf_kwargs = {
        "beta_loss": "frobenius",
        "init": "random",
        "l1_ratio": 0.0,
        "max_iter": 400,
        "solver": "cd",
        "tol": 0.0001,
        "n_components": 16,
        "update_H": False,
    }
    return nmf_kwargs


def apply_nmf(adata):
    """Apply nmf to hvgs in AnnData object, using saved gene loadings.
    Returns a pd.DataFrame of GEP usages for each cell.
    """
    df_std, df_loadings = load_scale_loadings()
    # find hvgs in adata
    is_hvg = adata.var_names.isin(df_loadings.index)
    X_hvg = adata.X[:, is_hvg].A
    hvg_names = adata.var_names[is_hvg]
    # rescale loadings
    df_loadings = df_loadings.loc[hvg_names]
    df_loadings /= df_loadings.sum(0)
    df_std = df_std.loc[hvg_names]
    # perform scaling by dividing TPT-normalized expression by gene sd
    if "round2" in adata.obs:
        # separate gene scaling for each batch
        label_round2 = LabelEncoder().fit_transform(adata.obs.round2)
        X_hvg = X_hvg / df_std.values.T[label_round2]
    else:
        # apply the mean of the two home-cage SDs
        X_hvg = X_hvg / df_std.values.mean(1)
    nmf_kwargs = make_kwargs()
    nmf_kwargs["H"] = df_loadings.T.values
    # get usages for each cell, keeping gene loadings/weights constant
    # refit NMF
    W, _, _ = non_negative_factorization(X_hvg, **nmf_kwargs)
    df_nmf = pd.DataFrame(W, columns=df_loadings.columns, index=adata.obs_names)
    return df_nmf
