import itertools

import h5py
from numba import njit
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import statsmodels.stats.multitest as smm
from tqdm import trange

from osn.preprocess import get_data_folders


def trace_filename(h5_name="imaging_preprocessed_traces.h5"):
    data_fold = get_data_folders()
    trace_file = data_fold.processed / h5_name
    return trace_file


def is_env(df, env_key="env"):
    uq_envs = df[env_key].unique()
    return {e: df[env_key] == e for e in uq_envs}


def load_df_meta():
    """Read pd.DataFrame with metadata for each trial"""
    df_meta = pd.read_hdf(trace_filename(), key="metadata")
    env_mapping = {k: is_env(df) for k, df in df_meta.groupby(level=0)}
    return df_meta, env_mapping


def load_key(key=None, asstr=False, fn=None):
    """Load key from h5 trace_filename"""
    if fn is None:
        fn = trace_filename()
    with h5py.File(fn, "r") as f:
        if key == "odors" or asstr:
            return f[key].asstr()[()]
        else:
            return f[key][()]


def load_traces(key=None, fn=None, base="data/traces"):
    """Load traces from h5 trace_filename"""
    if fn is None:
        fn = trace_filename()
    out_dict = {}
    with h5py.File(fn, "r") as f:
        traces = f[base]
        if key is None:
            # load all
            for k, v in traces.items():
                out_dict[k] = v[()]
        else:
            if isinstance(key, str):
                key = [key]
            keys_found = set(key) & traces.keys()
            if len(keys_found) > 0:
                print(f"Found keys {keys_found} in {traces.keys()}, loading...")
                for k in keys_found:
                    out_dict[k] = traces[k][()]
            else:
                print(f"Didn't find key {key} in {traces.keys()}")
    return out_dict


def z_score_stack(trace_stack):
    """Z-score each trial (single presentation of each odor) for each glom

    Args:
        trace_stack (np.array): odors x trials x glom x timepoints
    """
    N_ODORS, N_TRIALS, N_GLOM, NT = trace_stack.shape
    t_mean = np.nanmean(trace_stack, -1, keepdims=True).mean(0, keepdims=True)
    t_std = np.sqrt(
        np.nansum((trace_stack - t_mean) ** 2, axis=-1, keepdims=True).sum(
            0, keepdims=True
        )
        / (NT * N_ODORS)
    )
    all_z = (trace_stack - t_mean) / t_std
    return all_z


def baseline_stack(traces, x, start=-5, end=-0.2):
    pre_odor = (x > start) & (x <= end)
    return {
        k: v - v[:, :, :, pre_odor].mean(-1, keepdims=True) for k, v in traces.items()
    }


def odor_mean(trace_mat, x, pre=(-3.1, -0.1), post=(0, 3)):
    pre_odor = (x > pre[0]) & (x <= pre[1])
    odor = (x > post[0]) & (x <= post[1])
    pre_mean = trace_mat[:, :, :, pre_odor].mean(-1)
    post_mean = trace_mat[:, :, :, odor].mean(-1)
    odor_resp = post_mean - pre_mean
    return odor_resp


def resp_glom(
    traces,
    env_mapping,
    x,
    odors,
    dff_thresh=0.125,
    sig_thresh=1e-3,
    blank_thresh=2,
    n_env_sig=2,
    diff_kwargs={},
):
    resp_dict = {}
    for mouse, trace_mat in traces.items():
        N_ODORS, _, N_GLOM, _ = trace_mat.shape
        odor_resp = odor_mean(trace_mat, x, **diff_kwargs)
        align_sig_dict = {}
        # test for each env separately
        for k, v in env_mapping[mouse].items():
            this_diff = odor_resp[:, v]
            d_mean = this_diff.mean(1)
            # test each odor-glom pair
            ps = np.zeros((N_ODORS, N_GLOM))
            ps_correct = np.zeros_like(ps)
            for i in range(N_ODORS):
                for j in range(N_GLOM):
                    ps[i, j] = stats.wilcoxon(this_diff[i, :, j]).pvalue
                ps_correct[i, :] = smm.multipletests(ps[i, :], method="holm")[1]
            is_sig = (ps_correct <= sig_thresh) & (np.abs(d_mean) >= dff_thresh)
            align_sig_dict[k] = {
                "p": ps,
                "p_adj": ps_correct,
                "d_mean": d_mean,
                "inc": d_mean > 0,
                "sig": is_sig,
            }
        # glomeruli sig for both env
        sig_both = (
            np.stack([v["sig"] for v in align_sig_dict.values()]).sum(0) >= n_env_sig
        )
        blank_idx = np.where(odors == "blank")[0]
        # blank_thresh times higher response to odor than blank
        above = [
            np.abs(v["d_mean"]) / np.abs(v["d_mean"][blank_idx]) > blank_thresh
            for v in align_sig_dict.values()
        ]
        above_blank = np.stack(above).sum(0) > 0
        # sig and high enough response
        sig_keep = (sig_both) & (above_blank)
        # keep sig ones for blank
        sig_keep[blank_idx] = sig_both[blank_idx]
        resp_dict[mouse] = {"sig_keep": sig_keep, "sig_dict": align_sig_dict}
    return resp_dict


@njit
def nperm(n):
    return np.random.permutation(n)


def env_glom(
    traces,
    env_mapping,
    x,
    odors,
    resp_dict,
    nperms = 100_000,
    HOME = "home-cage",
    diff_kwargs={}):
    not_blank = odors != "blank"
    odor_keep = odors[not_blank]
    mean_traces = {}
    for k, trace_mat in traces.items():
        mean_traces[k] = odor_mean(trace_mat, x, **diff_kwargs)[not_blank]

    odor_mean_dict = {}
    sig_dict = {}

    for _mouse, v in mean_traces.items():
        sig_keep = resp_dict[_mouse]['sig_keep'][not_blank]
        sig_dict[_mouse] = sig_keep
        # concatenate resp glomeruli for each odor
        o_means = []
        for i, is_resp in enumerate(sig_keep):
            o_means.append(v[i][:, is_resp])
        o_means = np.concatenate(o_means, axis=1)
        odor_mean_dict[_mouse] = o_means

    sig_dfs = {}
    for _mouse, o_means in odor_mean_dict.items():
        print(f"Getting odor-glom pairs that differ across environments for mouse {_mouse}...")
        is_a_h = env_mapping[_mouse][HOME]
        n_a_h = is_a_h.sum()
        n_either, n_glom = o_means.shape
        obs = o_means[:n_a_h].mean(0) - o_means[n_a_h:].mean(0)
        obs_sign = np.sign(obs)
        abs_obs = np.abs(obs)

        sig_keep = sig_dict[_mouse]
        na_glom = np.arange(sig_keep.shape[1])
        xo = np.repeat(np.arange(sig_keep.shape[0]), sig_keep.sum(1))
        df_sig = pd.DataFrame(xo, columns=["odor"])
        df_sig["glom"] = np.hstack([na_glom[s] for s in sig_keep])
        df_sig["name"] = odor_keep[df_sig.odor]

        counts = np.zeros(n_glom)
        for i in trange(nperms - 1, ncols=100):
            perm = nperm(n_either)
            o_perm = o_means[perm]
            diff = o_perm[:n_a_h].mean(0) - o_perm[n_a_h:].mean(0)
            counts += (diff * obs_sign) > abs_obs

        emp_p = (counts + 1) / nperms
        emp_fdr = smm.multipletests(emp_p, method="fdr_bh")[1]

        df_sig["obs"] = obs
        df_sig["inc"] = obs > 0
        df_sig["p"] = emp_p
        df_sig["p_adj"] = emp_fdr
        sig_dfs[_mouse] = df_sig
    return sig_dfs



def corr_mat(traces, env_mapping, x, odors, resp_dict, diff_kwargs={}):
    not_blank = odors != "blank"
    mean_traces = {}
    for k, trace_mat in traces.items():
        mean_traces[k] = odor_mean(trace_mat, x, **diff_kwargs)[not_blank]
    tril_dict = {}
    for k, v in env_mapping.items():
        tril_dict[k] = {e: np.tril_indices(vv.sum(), -1) for e, vv in v.items()}
    N_ODORS = not_blank.sum()
    odor_keep = odors[not_blank]
    COMBOS = list(itertools.combinations_with_replacement(range(N_ODORS), 2))
    N_COMBOS = len(COMBOS)
    
    mouse_dfs = {}
    for _mouse, o_mean in mean_traces.items():
        idxs = tril_dict[_mouse]
        is_trial_env = env_mapping[_mouse]
        cmats = {k: np.zeros(N_COMBOS) for k in idxs.keys()}
        sig = resp_dict[_mouse]['sig_keep'][not_blank]
        # use union of resp glomeruli for each odor pair
        for n, (i, j) in enumerate(COMBOS):
            is_s1 = sig[i]
            is_s2 = sig[j]
            is_s12 = (is_s1) | (is_s2)
            o1 = o_mean[i][:, is_s12]
            o2 = o_mean[j][:, is_s12]
            # median of all trial pairs within same env (within and between odors)
            for k, v in is_trial_env.items():
                dmat = 1 - metrics.pairwise_distances(o1[v], o2[v], metric="correlation")
                if i == j:
                    # avoid double-counting trials and diagonal (same-trial)
                    cmats[k][n] = np.median(dmat[idxs[k]])
                else:
                    cmats[k][n] = np.median(dmat)
        # add odor labels and make symmetric
        dfs = {}
        for k, v in cmats.items():
            cmat = np.zeros((N_ODORS, N_ODORS))
            for (i, j), val in zip(COMBOS, v):
                cmat[i, j] = val
                cmat[j, i] = val
            dfs[k] = pd.DataFrame(cmat, index=odor_keep, columns=odor_keep)
        mouse_dfs[_mouse] = dfs
    return mouse_dfs