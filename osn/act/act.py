import numpy as np
import pandas as pd
import scanpy as sc

from osn.preprocess.util import get_data_folders
from osn.olfr import olfr


def z_scale(X, df_std):
    mu, sd = df_std.values.T
    return (X - mu) / (sd)


def load_scale_and_weights():
    """Load saved IEG and activation gene mean and SD for both v3 and v3.1 data.
    To use for z-scoring the IEGs and activation genes
    to identify activated cells and calculate the activation score, respectively"""
    data_fold = get_data_folders()
    df_std = pd.read_csv(
        data_fold.tables / "Act-seq_ieg_activation_gene_scaling.csv",
        index_col=[0, 1, 2],
    )
    # load gene loadings for each activation gene, to calculate the activation score
    df_loadings = pd.read_csv(
        data_fold.tables / "GSE173947_Table_S5_activation_genes.csv", index_col=0
    )
    return df_std, df_loadings


def dpg_olfr():
    """ORs activated via DPG 2h; likely from semiochemicals"""
    return ["Olfr1458", "Olfr910", "Olfr912"]


def apply_activation_score(adata, Z_thresh=None, platform="v3"):
    """Add ieg score and activation pca for each cell in AnnData.
    Activation pca is calculated as a product of the z-scored activation genes and saved weights for each gene, to be used for the activation score calculation.
    Updates AnnData object in place.

    Args:
        adata (AnnData)
        Z_thresh: threshold of z-scored IEGs to consider an OSN as activated
        platform: 10X genomics platform. Defaults to "v3". Data with v3.1 kit uses different `Z_thresh`
    """
    df_std_all, df_loadings = load_scale_and_weights()
    if platform not in ["v3", "v3.1"]:
        raise ValueError("Platform must be either 'v3' or 'v3.1'")
    thresh_map = {"v3": 0.2, "v3.1": -0.05}
    if Z_thresh is None:
        Z_thresh = thresh_map[platform]
    df_std = df_std_all.loc[platform]
    df_ieg = df_std.loc["ieg"]
    X_ieg = sc.get.obs_df(adata, list(df_ieg.index))
    X_ieg_scale = z_scale(X_ieg, df_ieg)

    # add ieg to adata
    adata.obs["n_ieg"] = (X_ieg > 0).sum(1)
    adata.obs["ieg_z_scored"] = X_ieg_scale.mean(1)
    adata.obs["activated"] = adata.obs["ieg_z_scored"] >= Z_thresh

    # add activation score
    act_genes = list(df_loadings.index)
    df_act = df_std.loc["act"].loc[act_genes]
    adata.obs["activation_pca"] = (
        z_scale(sc.get.obs_df(adata, act_genes), df_act)
        @ df_loadings.activation_score_loading
    )


def get_activated_olfrs(
    adata,
    df_OR,
    odors=None,
    DPG="DPG30m",
    pct_thresh=0.7,
    ctl_thresh=0.2,
    n_ieg_thresh=4,
    exclude_dpg_olfr=False,
    source_thresh=4,
):
    """Determine which ORs are activated, based on patterns of IEG expression.

    Args:
        adata (AnnData)
        df_OR (pd.DataFrame)
        odors: list of odors to consider
        DPG (str, optional): name of control condition. Defaults to "DPG30m".
        pct_thresh (float, optional): Minimum fraction of cells in odor activated per OR to be called as activated. Defaults to 0.7.
        ctl_thresh (float, optional): Maximum fraction of cells in DPG activated. ORs activated above threshold are not considered. Defaults to 0.2.
        n_ieg_thresh (int, optional): mean # of IEGs activated per OR to count as activated. Defaults to 4.
        exclude_dpg_olfr (bool, optional): Exclude ORs activated in DPG2h. Defaults to False.
        source_thresh (int, optional): OSNs per OR per condition to consider. Defaults to 4.
    Returns:
        odor_dict: dict of dict for each odor with keys:
        'active': boolean series of activated ORs
        'via_ieg': set of activated ORs
        'df': OR means for DPG/ODOR
        'delta': subtract DPG mean for each OR from the odor mean (e.g. `activation_score`, which is the delta of the `activation_pca`)
    """
    if not np.in1d(odors, df_OR.odor).all():
        raise ValueError(f"Odors {odors} are not all in {set(df_OR.odor)}")

    # add activation columns to df_OR
    col_add = pd.Index(["n_ieg", "ieg_z_scored", "activated", "activation_pca"])
    df_OR_act = df_OR.join(adata.obs[col_add.difference(df_OR.columns)])

    if exclude_dpg_olfr:
        # remove ORs activated in DPG
        df_OR_act = df_OR_act[~df_OR_act.top_Olfr.isin(dpg_olfr())]
    if "env" in df_OR_act:
        # remove outlier with promiscuous activation in env experiments
        df_OR_act = df_OR_act[df_OR_act.top_Olfr != "Olfr134"]

    odor_dict = {}
    for _odor in odors:
        odors_test = (DPG, _odor)
        df_subset = df_OR_act[df_OR_act.odor.isin(odors_test)]
        _, has_enough = olfr.filter_OR_source(df_subset, source_thresh, col="odor")
        df_subset_keep = df_subset[has_enough].copy()
        df_subset_mean = df_subset_keep.groupby(["odor", "top_Olfr"]).mean()
        uq_olfr = np.unique(df_subset_keep.top_Olfr)
        # activated in odor source but not in control
        # with enough IEGs activated per OR
        pcts = df_subset_mean["activated"]
        is_active = (
            (pcts.loc[_odor] >= pct_thresh)
            & (pcts.loc[DPG] <= ctl_thresh)
            & (df_subset_mean.n_ieg.loc[_odor] >= n_ieg_thresh)
        )
        via_ieg = is_active.index[is_active]
        print(f"Found {len(via_ieg)} ORs ({is_active.mean() * 100:.3f} %) for {_odor}")
        # only use activated cells for calculations
        to_remove = (
            (df_subset_keep.top_Olfr.isin(via_ieg))
            & (df_subset_keep.odor == _odor)
            & (~df_subset_keep.activated)
        )
        df_subset_filt = df_subset_keep[~to_remove]
        df_subset_mean = df_subset_filt.groupby(["odor", "top_Olfr"]).mean()
        # subtract at OR level mean of DPG condition from mean of odor condition
        # the `activation_score` columnn of `df_delta` then has the activation score for each OR
        df_delta = df_subset_mean.loc[_odor] - df_subset_mean.loc[DPG]
        df_delta["is_active"] = is_active
        df_delta["activation_score"] = df_delta["activation_pca"]
        df_subset_mean["is_active"] = df_subset_mean.index.get_level_values(1).isin(
            via_ieg
        )
        odor_dict[_odor] = {
            "active": is_active,
            "via_ieg": via_ieg,
            "df": df_subset_mean,
            "delta": df_delta,
        }
    return odor_dict