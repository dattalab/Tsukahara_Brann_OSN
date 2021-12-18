from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial
import scipy.cluster.hierarchy as sch
import seaborn as sns

from osn.olfr import olfr
from osn.viz.viz import update_cbar


def default_order():
    odor_order = [
        "Benzaldehyde",
        "Trans-2-Methyl-2-butenal",
        "Pentanal",
        "Hexanal",
        "Heptanal",
        "Butyl acetate",
        "Ethyl acetate",
        "Propyl acetate",
        "Butanone",
        "Pentanone",
        "Hexanone",
        "Propanal",
        "Butanal",
        "Methyl tiglate",
        "Ethyl butyrate",
        "Methyl valerate",
    ]
    return odor_order


def plot_pct(df, x="level_0", y="sig", hue="name", odor_order=None):
    if odor_order is None:
        odor_order = default_order()
    hsl_pal = col_pal(odor_order)
    fig, ax = plt.subplots(figsize=(3, 6))
    sns.pointplot(data=df, x=x, y=y, join=False, color="k", scale=1.5, zorder=10)
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        hue_order=odor_order,
        palette=hsl_pal,
        alpha=0.75,
        size=12,
        jitter=0.31,
        zorder=0,
    )
    ax.legend().remove()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Mouse")
    sns.despine()
    return fig, ax


def odor_color(fig, gs, coord, xon, hus_cmap, left=True):
    if left:
        to_plot = xon[:, None]
    else:
        to_plot = xon[None, :]
    ax = fig.add_subplot(
        gs[coord],
        xticks=[],
        yticks=[],
        frameon=False,
    )
    ax.imshow(
        to_plot,
        interpolation="none",
        cmap=hus_cmap,
        aspect="auto",
        rasterized=True,
    )
    return ax


def make_gs(width_ratios, height_ratios):
    fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
    gs = mpl.gridspec.GridSpec(
        len(height_ratios),
        len(width_ratios),
        fig,
        0.0,
        0.0,
        1,
        1,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0,
        hspace=0,
    )
    return fig, gs


def add_odor_names(left_ax, xon, odor_order):
    left_ax.set_yticks(xon)
    odor_names = deepcopy(odor_order)
    odor_names[odor_names == "Trans-2-Methyl-2-butenal"] = "Tiglaldehyde"
    left_ax.set_yticklabels(odor_names)
    left_ax.yaxis.set_tick_params(pad=1)


def col_pal(odor_order, pal="husl"):
    return sns.color_palette(palette=pal, n_colors=len(odor_order))


def odor_pal(odor_order):
    N_ODORS = len(odor_order)
    hsl_pal = col_pal(odor_order)
    xon = np.arange(N_ODORS)
    hus_cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", hsl_pal)
    return xon, hus_cmap


def plot_corr(dfs, HOME="home-cage", eA="envA", metric="cityblock", method="ward"):
    df_home = dfs[HOME]
    cl_dist = spatial.distance.pdist(df_home, metric=metric)
    cl_link = sch.linkage(cl_dist, method)
    cl_leaves_order = sch.leaves_list(cl_link)
    odor_keep = df_home.index.values
    odor_order = odor_keep[cl_leaves_order]
    xon, hus_cmap = odor_pal(odor_order)

    width_ratios = [0.36, 5.2, 0.24, 0.36, 5.2]
    height_ratios = [5.2, 0.46]
    fig, gs = make_gs(width_ratios, height_ratios)

    for col in [3, 0]:
        left_ax = odor_color(fig, gs, (0, col), xon, hus_cmap, left=True)
    add_odor_names(left_ax, xon, odor_order)

    for col, env in zip([1, 4], (HOME, eA)):
        ax = odor_color(fig, gs, (1, col), xon, hus_cmap, left=False)
        dist_ax = fig.add_subplot(gs[0, col], yticks=[], xticks=[], frameon=False)
        dist_im = dist_ax.imshow(
            dfs[env].loc[odor_order, :][odor_order],
            vmin=-1,
            vmax=1,
            cmap="PRGn",
            rasterized=True,
        )
        dist_ax.set_title(env)

    cb_ax = fig.add_axes([1.02, 0.05, 0.03, 0.9])
    cbar = plt.colorbar(dist_im, cax=cb_ax)
    cb_ax.yaxis.set_tick_params(pad=1)
    cbar.set_label("Population vector correlation ($R$)")
    update_cbar(cbar)

    return fig, odor_order


def plot_delta(dfs, odor_order, HOME="home-cage", eA="envA", vmin=-0.4, vmax=0.4):
    xon, hus_cmap = odor_pal(odor_order)
    df_delta = dfs[eA] - dfs[HOME]

    width_ratios = [0.36, 5.2]
    height_ratios = [5.2, 0.36]
    fig, gs = make_gs(width_ratios, height_ratios)

    dist_ax = fig.add_subplot(
        gs[0, 1],
        yticks=[],
        xticks=[],
        frameon=False,
    )
    dist_im = dist_ax.imshow(
        df_delta.loc[odor_order, :][odor_order],
        vmin=vmin,
        vmax=vmax,
        cmap="cmo.balance",
        rasterized=True,
    )
    dist_ax.set_title(f"{eA} - {HOME}")

    left_ax = odor_color(fig, gs, (0, 0), xon, hus_cmap, left=True)
    add_odor_names(left_ax, xon, odor_order)
    odor_color(fig, gs, (1, 1), xon, hus_cmap, left=False)

    cb_ax = fig.add_axes([1.05, 0.05, 0.06, 0.9])
    cbar = plt.colorbar(dist_im, cax=cb_ax)
    cbar.set_label("Change in population\n" + r"vector correlation ($R$)")
    update_cbar(cbar)

    return fig


def plot_env_decoding(df_combos, odor_order=None):
    if odor_order is None:
        odor_order = default_order()
    
    fig, ax = plt.subplots(figsize=(6, 6))

    y_obs = df_combos[~df_combos.shuff].pivot(index="n_glom", columns="odor", values="value")
    y_shuff = df_combos[df_combos.shuff].pivot(index="n_glom", columns="odor", values="value")[odor_order]

    x = np.unique(df_combos.n_glom)
    hsl_pal = col_pal(odor_order)

    for c, h in zip(hsl_pal, odor_order):
        ax.plot(x, y_obs[h], lw=1.5, color=c, alpha=0.85);
    #     ax.plot(x, y_shuff[h], lw=0.5, color=c, alpha=0.5);
    ax.plot(x, y_shuff, lw=1.5, color="0.7", alpha=0.5, ls="--");

    sns.despine()
    ax.set_xticks(np.arange(x.max()+1, step=10))
    ax.set_xlim(0, x.max())
    ax.set_ylim(None, 100)
    ax.set_ylabel("Environment classification accuracy (%)")

    ax.text(50, 50.5, "shuffled", ha="right", va="bottom")
    ax.text(50, 100, "observed", ha="right", va="bottom")
    ax.set_xlabel("# of glomeruli")
    return fig, ax


def plot_odor_decoding(obs_melt, shuff_melt, n_odors=16):
    xx = np.unique(obs_melt.n_glom)
    df_shuff = pd.concat(
        {
            "a": obs_melt[obs_melt.kind == "shuffled"],
            "b": shuff_melt[shuff_melt.kind == "shuffled"],
        }
    ).reset_index()
    obs_melt = obs_melt[obs_melt.kind == "observed"]
    shuff_env_melt = shuff_melt[shuff_melt.kind == "observed"]
    means = obs_melt.groupby(["variable", "n_glom"]).v100.mean()
    shuff_env_means = shuff_env_melt.groupby(["n_glom"]).v100.mean()
    shuff_means = df_shuff.groupby(["level_0", "variable", "n_glom"]).v100.mean()
    iqrs = obs_melt.groupby(["variable", "n_glom"]).v100.apply(olfr.get_ci_vect_vectorized)

    order = ["home-cage: home-cage", "envA: envA", "home-cage: envA", "envA: home-cage"]
    labels = ["within env", "within env", "between env", "between env"]
    cls = [plt.cm.tab10(i) for i in [0, 0, 2, 2]]
    mapping = dict(zip(order, cls))
    lab_mapping = dict(zip(order, labels))

    fig, ax = plt.subplots(figsize=(6, 6))
    for k, v in mapping.items():
        ax.plot(xx, means.loc[k], color=v, lw=1.5, label=lab_mapping[k])
        cis = np.stack(iqrs.loc[k].values)
        ax.fill_between(xx, *cis.T, alpha=0.75, lw=0, color=v)
    ax.plot(
        xx,
        shuff_env_means,
        color=plt.cm.tab10(3),
        lw=1.5,
        label="shuffled env",
        ls="--",
        zorder=10,
    )
    ax.plot(xx, shuff_means.unstack().T, color="0.7", lw=1.5, ls="--")
    ax.text(50, 100 / n_odors + 1, "shuffled odor", ha="right", va="bottom")
    ax.set_xticks(np.arange(xx.max() + 1, step=10))
    ax.set_xlim(4, 50)
    ax.set_ylim(0, 102)
    ax.set_xlabel("# of glomeruli")
    ax.set_ylabel("Odor classification accuracy (%)")
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[::2],
        labels[::2],
        handlelength=1,
        borderaxespad=0,
        labelspacing=0.25,
        handletextpad=0.4,
        loc="upper right",
        bbox_to_anchor=(1, 0.75),
    )
    for lh in leg.legendHandles:
        lh.set_linewidth(2)
    sns.despine()
    return fig, ax
