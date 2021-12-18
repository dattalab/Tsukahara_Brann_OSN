import itertools

import numpy as np
import numpy_groupies as npg
import pandas as pd
import statsmodels.stats.multitest as smm
from sklearn.preprocessing import LabelEncoder

from osn.olfr.olfr import filter_OR_source


def dprime(act, or_labels, env):
    """Calculate d-prime statistics (difference in means divided by standard deviation)"""
    gmat = np.vstack((or_labels, env))
    mean_diff = np.diff(npg.aggregate(gmat, act, func="mean"), axis=1).flatten()
    or_var = npg.aggregate(gmat, act, func="var")
    denom = np.sqrt(or_var.mean(1))
    return mean_diff / denom


def get_dprime(
    df_dpg, ENV_USE, within=False, env_col="env", n_perm=10000, seed=12345, thresh=4
):
    """Calculate d-prime statistic for ES score difference across ENVS for each OR

    Args:
        df_dpg: df_OR-like pd.DataFrame
        ENV_USE: list of envs
        within: if True, compare across mice within envs. if False, compare across env pairs

    Returns:
        dprime_combo_dict: dictionary with keys env pairs and results of d-prime calculations.
        p-values for each OR are found in `['p']`

    """
    env_combos = list(itertools.combinations(ENV_USE, 2))
    if within:
        col = "orig_ident"
        to_iter = df_dpg.groupby(env_col).orig_ident.unique().loc[ENV_USE].to_list()
        to_iter = [tuple(c) for c in to_iter]
    else:
        col = env_col
        to_iter = env_combos

    dprime_combo_dict = {}
    for combo in to_iter:
        df_combo = df_dpg[df_dpg[col].isin(combo)]
        e1, e2 = combo
        # enough cells in both sources
        _, has_enough = filter_OR_source(df_combo, thresh, col=col)
        df_dpg_keep = df_combo[has_enough].copy()
        df_dpg_mean = df_dpg_keep.groupby(["top_Olfr", col]).mean()
        dpg_act = df_dpg_keep.ES_score.values
        df_dpg_act_mean = df_dpg_mean["ES_score"].unstack()
        es_diff = df_dpg_act_mean[e2] - df_dpg_act_mean[e1]
        uq_olfr = np.unique(df_dpg_keep.top_Olfr)
        n_olfr = len(uq_olfr)
        print(f"Using {n_olfr} ORs for combo: {combo}")
        or_labels = LabelEncoder().fit_transform(df_dpg_keep.top_Olfr)
        is_e2 = (df_dpg_keep[col] == e2).astype(int).values
        # make permutated env matrix for each olfr
        np.random.seed(seed)
        perm = np.zeros((len(df_dpg_keep), n_perm), dtype=np.int)
        for i, olfr in enumerate(uq_olfr):
            is_olfr = df_dpg_keep.top_Olfr == olfr
            perm_mat = np.random.rand(is_olfr.sum(), n_perm).argsort(0)
            perm[is_olfr, :] = is_e2[is_olfr][perm_mat]
        # observed
        d_prime = dprime(dpg_act, or_labels, is_e2)
        # shuffled
        perms = []
        for i in range(n_perm):
            perms.append(dprime(dpg_act, or_labels, perm[:, i]))
        d_perm = np.stack(perms)
        # one-sided empirical p-value
        p1 = (d_perm <= d_prime).mean(0)
        p2 = (d_perm > d_prime).mean(0)
        p_min = np.min([p1, p2], axis=0)
        p_adj = smm.multipletests(p_min, method="fdr_bh")[1]
        # summarize d-prime results into dataframe
        df_p = pd.DataFrame(
            [p1, p2, p_min, p_adj, d_prime],
            columns=uq_olfr,
            index=["less", "greater", "p", "p_adj", "d_prime"],
        ).T.join(df_dpg_act_mean)
        df_p["delta_ES_score"] = es_diff
        df_p["sig"] = df_p.p_adj <= 0.01
        df_p["pos"] = df_p.d_prime > 0
        print(f"{np.mean(df_p.sig) * 100:.3f} % of ORs have significant d-prime values")
        sig_pos = df_p.sig.map({False: "non-sig", True: "sig, dec"}).values
        sig_pos[(df_p.sig) & (df_p.pos)] = "sig, inc"
        df_p["sig_pos"] = sig_pos
        dprime_combo_dict[combo] = {
            "olfr": uq_olfr,
            "p": df_p,
            "df": df_dpg_keep,
            "sig": df_p.index[df_p.sig],
        }
    return dprime_combo_dict