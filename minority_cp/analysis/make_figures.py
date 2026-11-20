"""Turn results/*.csv into figures/*.png.   Run:  python -m analysis.make_figures"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.common import RESULTS, FIGURES, ALPHA

C = ["#1F2A44", "#0F8B8D", "#E0A030", "#B5483A", "#7A8AA8"]


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", FIGURES / name)


def fig_min_size():                                   # RQ1
    d = read("min_size.csv")
    if d is None: return
    d = d[d.finite_share > 0.99]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.n_cal, d.cov_sd, "o-", color=C[1], label="simulated")
    ax.plot(d.n_cal, d.theory_sd, "--", color=C[0], label="Beta theory")
    ax.axvspan(0, 8.5, color="#E3E7EE", label="no finite threshold (n < 9)")
    ax.set_xscale("log"); ax.set_xlim(3, 100)
    ax.set_xlabel("rare calibration points (n)"); ax.set_ylabel("std. dev. of coverage")
    ax.set_title(f"Coverage noise vs calibration size (alpha = {ALPHA})"); ax.legend()
    save(fig, "fig1_min_size.png")


def fig_regime_map():                                 # RQ2
    d = read("regime_map.csv")
    if d is None: return
    models = list(d.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        t = d[d.model == name].pivot(index="m", columns="lam", values="set_size")
        delta = t.sub(t[0], axis=0).drop(columns=0)
        im = ax.imshow(delta.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_xticks(range(delta.shape[1])); ax.set_xticklabels(delta.columns)
        ax.set_yticks(range(delta.shape[0])); ax.set_yticklabels(delta.index)
        ax.set_xlabel("synthetic points (lambda)"); ax.set_ylabel("real rare points (m)")
        ax.set_title(f"{name}: change in set size")
        for (i, j), v in np.ndenumerate(delta.values):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=axes, label="blue = smaller sets (helps), red = larger (hurts)")
    save(fig, "fig2_regime_map.png")


def fig_leakage():                                    # RQ3
    d = read("leakage.csv")
    if d is None: return
    models = list(d.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        for c, mode in zip(C, d["mode"].unique()):
            s = d[(d.model == name) & (d["mode"] == mode)]
            ax.plot(range(len(s)), s.rare_cov, "o-", color=c, label=mode.replace("_", " "))
        lams = sorted(d.lam.unique())
        ax.set_xticks(range(len(lams))); ax.set_xticklabels(lams)
        ax.axhline(1 - ALPHA, color="k", ls=":", lw=1)
        ax.set_xlabel("synthetic points (lambda)"); ax.set_title(name)
    axes[0].set_ylabel("rare-class coverage (target 0.90)"); axes[0].legend(fontsize=8)
    save(fig, "fig3_leakage.png")


def grouped_bars(ax, d, col, groups, key, labels=None):
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        s = d[d[key] == g]
        ax.bar(np.arange(len(s)) + i * w, s[col], w, color=C[i % len(C)], label=g)
    ax.set_xticks(np.arange(len(s)) + 0.4 - w / 2)
    return s


def fig_rule():                                       # RQ4
    d = read("rule_vs_baselines.csv")
    if d is None: return
    methods = list(d.method.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["set_size", "reliable"],
                              ["average set size (lower = better)",
                               "share of runs with rare coverage >= 0.85"]):
        s = grouped_bars(ax, d, col, methods, "method")
        ax.set_xticklabels([f"m = {m}" for m in s.m]); ax.set_title(title)
        if col == "set_size": ax.set_ylim(min(1.0, d[col].min() - 0.1), d[col].max() + 0.05)
    axes[0].legend(fontsize=7, loc="upper right")
    save(fig, "fig4_rule_vs_baselines.png")


def fig_small_m():                                    # RQ5
    d = read("cross_conformal.csv")
    if d is None: return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, title in zip(axes, ["cov_sd", "worst_cov", "set_size"],
                              ["coverage spread (lower = steadier)",
                               "worst single run (higher = safer)", "average set size"]):
        for c, meth in zip(C, d.method.unique()):
            s = d[d.method == meth]
            ax.plot(s.m, s[col], "o-", color=c, label=meth)
        ax.set_xlabel("real rare points (m)"); ax.set_title(title)
    axes[0].legend(fontsize=8)
    save(fig, "fig5_small_m.png")


def fig_real():                                       # real datasets
    d = read("real_datasets.csv")
    if d is None: return
    sets = list(d.dataset.unique())
    fig, axes = plt.subplots(2, len(sets), figsize=(min(5.4 * len(sets), 21), 6.4), squeeze=False)
    for j, name in enumerate(sets):
        s = d[d.dataset == name].reset_index(drop=True)
        y = np.arange(len(s))
        ax = axes[0, j]
        ax.barh(y, s.rare_cov, color=C[1]); ax.axvline(1 - ALPHA, color="k", ls=":")
        for yi, v in zip(y, s.rare_cov): ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
        ax.set_title(f"{name} (m = {s.m[0]})"); ax.set_xlim(0, 1.15)
        ax.set_xlabel("rare-class coverage (target 0.90)")
        ax.set_yticks(y); ax.set_yticklabels(s.method if j == 0 else [""] * len(s)); ax.invert_yaxis()
        # set-size comparison: marginal CP is left out because its small sets are mostly empty sets
        t = s[~s.method.str.startswith("Marginal")].reset_index(drop=True); yt = np.arange(len(t))
        ax = axes[1, j]
        ax.barh(yt, t.size_vs_plain, xerr=t.ci95, color=C[2]); ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("set size vs Mondrian 50/50 plain (95% CI, left = smaller)")
        ax.set_yticks(yt); ax.set_yticklabels(t.method if j == 0 else [""] * len(t)); ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig6_real_datasets.png")


def fig_real_regime(csv="real_regime.csv", out="fig7_real_regime_map.png"):   # real-data regime map
    d = read(csv)
    if d is None: return
    sets, models = list(d.dataset.unique()), list(d.model.unique())
    fig, axes = plt.subplots(len(sets), len(models), figsize=(5 * len(models), 3.7 * len(sets)),
                             constrained_layout=True, squeeze=False)
    for i, name in enumerate(sets):
        for j, model in enumerate(models):
            s = d[(d.dataset == name) & (d.model == model) & (d.lam > 0)]
            piv = s.pivot(index="m", columns="lam", values="size_vs_lam0")
            ci = s.pivot(index="m", columns="lam", values="ci95")
            ax = axes[i, j]
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
            ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
            ax.set_xlabel("synthetic points (lambda)"); ax.set_ylabel("real rare points (m)")
            ax.set_title(f"{name}, {model}")
            for (a, b), v in np.ndenumerate(piv.values):
                star = "*" if abs(v) > ci.values[a, b] else ""
                ax.text(b, a, f"{v:+.2f}{star}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=axes, label="change in set size vs no rebalancing (blue = smaller); * = 95% interval excludes 0")
    save(fig, out)


def run():
    for f in (fig_min_size, fig_regime_map, fig_leakage, fig_rule, fig_small_m, fig_real, fig_real_regime):
        f()
    fig_real_regime("real_regime_jitter.csv", "fig8_real_regime_ties_broken.png")


if __name__ == "__main__":
    run()
