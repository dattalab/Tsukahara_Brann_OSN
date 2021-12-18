import cmocean
import matplotlib as mpl
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from osn.olfr.olfr import get_ci_vect_vectorized


def plt_cbar(fig, scatter, pos=[0.2, 0.88, 0.6, 0.05], lab=None):
    cb_ax = fig.add_axes(pos)
    cbar = fig.colorbar(scatter, cax=cb_ax, orientation="horizontal")
    cb_ax.xaxis.set_label_position("top")
    cb_ax.xaxis.set_ticks_position("top")
    cb_ax.xaxis.set_tick_params(pad=0)
    cbar.set_label(lab)
    update_cbar(cbar)
    return cbar, cb_ax


def update_cbar(cbar, lw=None):
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    if lw is None:
        cbar.outline.set_visible(False)
    else:
        cbar.outline.set_linewidth(lw)


def update_boxen(ax, cls="k", lw=1.5, ls="--"):
    """Update lines and points in seaborn boxenplot

    Parameters
    ----------
    ax : matplotlib.axes
        Axes containing a boxenplot
    """
    for a in ax.lines:
        a.set_color(cls)
        a.set_linewidth(lw)
        a.set_linestyle(ls)
        a.set_alpha(1)
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
        else:
            # remove outlier points
            a.set_alpha(0)


def offest_pointplot(ax, offset=0.3):
    """Shift pointplot (when plotted with e.g. strip/swarmplot)"""
    offset = mpl.transforms.ScaledTranslation(offset, 0, ax.figure.dpi_scale_trans)

    for coll in ax.collections:
        trans = coll.get_transform()
        coll.set_transform(trans + offset)
    # shift everything else:
    for line in ax.lines:
        trans = line.get_transform()
        line.set_transform(trans + offset)


def adjust_lims(
    ax, xmin=None, xmax=None, cls="k", zo=-3, alpha=0.5, add_line=True, **kwargs
):
    """Match limits for x and y and add x-y line on the diagonal."""
    vlim = ax.axes.viewLim.extents
    if xmin is None:
        xmin = np.min(vlim[:2])
    if xmax is None:
        xmax = np.max(vlim[2:])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    if add_line:
        ax.plot(
            [xmin, xmax],
            [xmin, xmax],
            color=cls,
            ls="--",
            zorder=zo,
            alpha=alpha,
            **kwargs,
        )


def plot_genes(
    genes,
    df_hvg_mean,
    rc=(2, 3),
    fs=(15, 10),
    vert=False,
    plot_corr=False,
    xy=(0.5, None),
):
    fig, axes = plt.subplots(*rc, figsize=fs, sharex=True)
    if vert:
        ax_flat = axes.T.flatten()
    else:
        ax_flat = axes.flatten()
    x = df_hvg_mean.ES_score
    for ax, gn in zip(ax_flat, genes):
        y = df_hvg_mean[gn]
        ax.scatter(x, y, alpha=0.5, lw=0, s=25)
        r, pval = stats.spearmanr(x, y)
        ax.set_title(rf"{gn}")
        lims = y.max() - y.min()
        ax.set_ylim(0 - lims / 20, None)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, integer=True))
        if plot_corr:
            r, p = stats.pearsonr(x, y)
            ax.text(
                1,
                0.02,
                rf"$\rho$: {r:.2f}",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
                size=18,
            )
    sns.despine()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # https://stackoverflow.com/a/44020303
    ax = axes[0, 0]
    ax.yaxis.label.set_transform(
        mtransforms.blended_transform_factory(
            mtransforms.IdentityTransform(), fig.transFigure  # specify x, y transform
        )
    )  # changed from default blend (IdentityTransform(), a[0].transAxes)
    ax.yaxis.label.set_position((0.02, 0.5))
    ax.set_ylabel("Normalized expression")
    ax = axes[-1, 0]
    ax.xaxis.label.set_transform(
        mtransforms.blended_transform_factory(
            fig.transFigure,
            mtransforms.IdentityTransform(),  # specify x, y transform
        )
    )  # changed from default blend (IdentityTransform(), a[0].transAxes)
    ax.xaxis.label.set_position((0.5, 0.02))
    ax.set_xlabel("ES score")
    return fig, axes


def plot_r(ax, x, y, ha="right", va="bottom", r_loc=(1, 0.02)):
    r, pval = stats.spearmanr(x, y)
    ax.text(
        *r_loc,
        rf"$\rho$:{r:.3f}",
        horizontalalignment=ha,
        verticalalignment=va,
        transform=ax.transAxes,
    )


def plot_combos(
    mice,
    combo_dict,
    y="ES_score",
    log=False,
    fs=None,
    subset=None,
    color=None,
    **kwargs,
):
    """Plot lower diagonal for all pairs of a given combo"""
    n = len(mice) - 1
    if fs is None:
        fs = (n * 3, n * 3)

    fig, axes = plt.subplots(n, n, figsize=fs, sharex=True, sharey=True)

    for i, m1 in enumerate(mice[:-1]):
        for ax, m2 in zip(axes[i:, i], mice[(i + 1) :]):
            combo = (m1, m2)
            to_plot = combo_dict[combo][y].unstack()
            if log:
                to_plot = np.log(to_plot)
            if subset is not None:
                is_subset = to_plot.index.isin(subset)
                to_plot = to_plot[is_subset]

            if color is not None:
                c = combo_dict[combo][color].unstack()[m1].values
                cmax = np.percentile(np.abs(c), 97.5)
                ax.scatter(
                    to_plot[m1],
                    to_plot[m2],
                    c=c,
                    vmin=-1 * cmax,
                    vmax=cmax,
                    cmap=cmocean.tools.crop_by_percent(cmocean.cm.balance, 7),
                    alpha=1,
                    s=30,
                    lw=0,
                )
            else:
                ax.scatter(to_plot[m1], to_plot[m2], alpha=0.5, s=16)
            plot_r(ax, to_plot[m1], to_plot[m2], **kwargs)

    for ax, m2 in zip(axes[:, 0], mice[1:]):
        ax.set_ylabel(m2.replace("baseline", "homecage"))
    for ax, m1 in zip(axes[-1, :], mice[:-1]):
        ax.set_xlabel(m1.replace("baseline", "homecage"))
    sns.despine()
    is_tril = np.tril(np.ones(n, dtype=np.bool))
    for ax in axes[~is_tril]:
        ax.axis("off")
    for ax in axes[is_tril]:
        adjust_lims(ax)
    fig.tight_layout()
    return fig, axes


def plot_es_score_decoding(df_out, ENVS):
    cls2 = [
        plt.cm.Paired.colors[9],
        plt.cm.Paired.colors[7],
        "#1f4d22",
    ]
    col_mapping = dict(zip(ENVS, cls2))
    gb = df_out.groupby(["kind", "variable", "subset"])
    iqrs = gb.value.quantile([0.25, 0.5, 0.75]).unstack()
    means = gb.value.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    to_test = np.unique(df_out.subset)
    for k, v in col_mapping.items():
        df_k = df_out[df_out.variable == k]
        for kind, ls in zip(["observed", "shuffled"], ["-", "--"]):
            ax.plot(to_test, means.loc[(kind, k)], color=v, lw=2, label=k, ls=ls)
            iqr_subset = iqrs.loc[(kind, k)]
            ax.fill_between(
                to_test, iqr_subset[0.25], iqr_subset[0.75], alpha=0.25, lw=0, color=v
            )
    leg = ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(
        handles[::2],
        labels[::2],
        handlelength=1,
        loc="center right",
        bbox_to_anchor=(1.04, 0.7),
        borderaxespad=0,
        labelspacing=0.25,
        handletextpad=0.4,
        frameon=False,
    )
    for lh in leg1.legendHandles:
        lh.set_linewidth(2)
    ax.add_artist(leg1)
    ax.text(45, 36, "shuffled", ha="right", va="bottom")
    ax.text(45, 100, "observed", ha="right", va="bottom")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Environment classification accuracy (%)")
    ax.set_xlabel("# of ORs")
    ax.set_xlim(0, 45)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    sns.despine()
    return fig, ax


def ace_2ha_activation_score(
    df_OR,
    example_olfr=("Olfr160", "Olfr923"),
    names=None,
    DPG="DPG30m",
    order=["ACE0_01p", "ACE0_1p", "2HA0_01p", "2HA0_1p"],
    pcts=["0.01", "0.1", "0.01", "0.1"],
):
    if names is None:
        names = example_olfr
    df_subset = df_OR[(df_OR.top_Olfr.isin(example_olfr))]
    df_subset_mean = df_subset.groupby(["odor", "top_Olfr"]).mean()
    df_subset_mhx = df_subset[df_subset.odor != DPG].copy()
    dpg_mean = df_subset_mean["activation_pca"].loc[DPG]
    df_subset_mhx["activation"] = (
        df_subset_mhx.activation_pca - dpg_mean.loc[df_subset_mhx.top_Olfr].values
    )
    cat_cols = df_subset_mhx.columns[df_subset_mhx.dtypes == "category"]
    df_subset_mhx[cat_cols] = df_subset_mhx[cat_cols].astype(str)
    df_mean = df_subset_mhx.groupby(["top_Olfr", "odor"]).activation.mean()
    df_ci = df_subset_mhx.groupby(["top_Olfr", "odor"]).activation.apply(
        get_ci_vect_vectorized
    )
    fig, axes = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
    x = [0, 1, 2.5, 3.5]
    cols = [plt.cm.tab20(i) for i in [5, 4, 9, 8]]
    for olfr, ax, nm in zip(example_olfr, axes, names):
        ax.bar(x, df_mean.loc[olfr].loc[order], color=cols, lw=0, width=0.9, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(pcts, rotation=90)
        cis = np.stack(df_ci.loc[olfr].loc[order].values)
        ax.plot(np.array([x, x]), cis.T, color="k", lw=2, solid_capstyle="butt")
        ax.set_title(nm)
        ax.text(0.25, -0.2, "ACE", ha="center", transform=ax.transAxes)
        ax.text(0.75, -0.2, "2-HA", ha="center", transform=ax.transAxes)

    axes[0].set_ylabel("Activation score")
    axes[1].text(1.04, -0.08, "(%)", ha="center", transform=ax.transAxes)
    sns.despine()

    return fig, axes
