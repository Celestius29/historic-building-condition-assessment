"""Regenerate figures 1 to 4 from results/metrics/*.csv and write them to results/figures.

Run from anywhere: python src/figures/make_figures.py

Plotting code copied from notebooks/01_crack_segmentation_deepcrack.ipynb. Two changes: paths are resolved
relative to the repository root, and the tick-label rotation that the notebook's final Fig 2 and Fig 4 cells
took from an undefined helper (fix_ticks) is written inline.

Not regenerated here:
- Fig 5 and Fig 6 need test-set predictions, which need model checkpoints (not in the repository). The plotting
  functions fig5_gt_vs_pred_indices and fig6_qualitative_error_maps are provided for use after the notebook has
  produced predictions.
- Fig 7 (pipeline overview) was drawn separately; its code is not in the notebook.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
FIG, MET = str(REPO / "results" / "figures"), str(REPO / "results" / "metrics")
os.makedirs(FIG, exist_ok=True)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#666666"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "grid.linewidth": .5, "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "pdf.fonttype": 42})
LAB = {"bce_dice": "BCE+Dice", "bce_dice_cldice": "BCE+Dice+clDice"}


def save(name):
    plt.savefig(f"{FIG}/{name}.png"); plt.savefig(f"{FIG}/{name}.pdf")
    if matplotlib.get_backend().lower() != "agg": plt.show()
    plt.close()


def panel(ax, s): ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold")


def fig1_segmentation_comparison():
    d = pd.read_csv(f"{MET}/segmentation_3seeds.csv")
    dl = pd.read_csv(f"{MET}/segmentation_baselines.csv").query("arch == 'DeepLabV3Plus'")
    cfg = [("Unet", "bce_dice", "U-Net"), ("Unet", "bce_dice_cldice", "U-Net + clDice"), ("UnetPlusPlus", "bce_dice", "U-Net++")]
    cols = [OI["blue"], OI["green"], OI["orange"]]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9))
    for ax, m, t in zip(axs, ["test_iou", "test_dice"], ["Test IoU", "Test Dice"]):
        for i, (a, l, n) in enumerate(cfg):
            v = d[(d.arch == a) & (d.loss == l)][m].values
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="o", color=cols[i], capsize=4, ms=6, zorder=3)
            ax.scatter(i + np.linspace(-.12, .12, len(v)), v, color="k", s=8, zorder=4)
        ax.scatter([3], dl[m].values, facecolors="none", edgecolors=OI["grey"], s=45, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cfg] + ["DeepLabV3+\n(1 seed)"], rotation=25, ha="right")
        ax.set_ylabel(t); ax.set_xlim(-.6, 3.6)
    panel(axs[0], "(a)"); panel(axs[1], "(b)"); save("fig1_segmentation_comparison")


def fig2_index_fidelity():
    f = pd.read_csv(f"{MET}/index_fidelity.csv").query("arch == 'Unet'")
    order = ["length_px", "mean_width_px", "area_ratio", "n_branches", "n_junctions", "tortuosity"]
    NAME = {"length_px": "Length", "mean_width_px": "Mean width", "area_ratio": "Area ratio",
            "n_branches": "Branches*", "n_junctions": "Junctions*", "tortuosity": "Tortuosity"}
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, col, yl in zip(axs, ["spearman", "med_rel_err"], ["Spearman rho (predicted vs GT)", "Median relative error"]):
        for j, (l, c) in enumerate(zip(["bce_dice", "bce_dice_cldice"], [OI["blue"], OI["green"]])):
            g = f[f.loss == l].groupby("index")[col].agg(["mean", "std"]).loc[order]
            ax.errorbar(np.arange(len(order)) + (j - .5) * .28, g["mean"], yerr=g["std"], fmt="o", color=c, capsize=2, label=LAB[l])
        ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[o] for o in order])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); ax.set_ylabel(yl)
    axs[0].set_ylim(0, 1); axs[0].legend(frameon=False, loc="lower left"); axs[1].axhline(0, color="k", lw=.8)
    fig.supxlabel("* skeletons not pruned (see Fig 3). Mean ± std over 3 seeds, test set.", fontsize=7)
    panel(axs[0], "(a)"); panel(axs[1], "(b)"); save("fig2_index_fidelity")


def fig3_spur_pruning():
    r = pd.read_csv(f"{MET}/branch_pruning.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    for ax, k, t in zip(axs, ["n_branches", "n_junctions"], ["Branches", "Junctions"]):
        for l, c in zip(["bce_dice", "bce_dice_cldice"], [OI["blue"], OI["green"]]):
            g = r[(r["index"] == k) & (r.loss == l)].groupby("min_len")["med_rel_err"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", color=c, capsize=2, label=LAB[l])
        ax.axhline(0, color="k", lw=.8); ax.set_xticks([0, 10, 20]); ax.set_xlabel("Spur pruning length (px)"); ax.set_title(t)
    axs[0].set_ylabel("Median relative error (pred vs GT)"); axs[0].legend(frameon=False)
    panel(axs[0], "(a)"); panel(axs[1], "(b)"); save("fig3_spur_pruning")


def fig4_calibration_null_result():
    c = pd.read_csv(f"{MET}/index_calibration.csv").query("loss == 'bce_dice_cldice'")
    nm = {"area_ratio": "Area ratio", "density": "Length density", "mean_width_px": "Mean width"}
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, (raw, cal), yl, tt in zip(axs, [("bias_raw", "bias_cal"), ("spread_raw", "spread_cal")],
            ["Median relative error", "Spread of rel. error (5th-95th pct)"], ["Bias", "Per-image spread"]):
        for j, (col, lab, cc, mk_) in enumerate([(raw, "Raw", OI["red"], "o"), (cal, "Calibrated", OI["blue"], "s")]):
            g = c.groupby("index")[col].agg(["mean", "std"]).loc[list(nm)]
            ax.errorbar(np.arange(3) + (j - .5) * .2, g["mean"], yerr=g["std"], fmt=mk_, color=cc, capsize=2, label=lab)
        ax.set_xticks(range(3)); ax.set_xticklabels(nm.values())
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); ax.set_ylabel(yl); ax.set_title(tt)
    axs[0].axhline(0, color="k", lw=.8); axs[0].legend(frameon=False, loc="lower left")
    fig.suptitle("A global scale factor fitted on validation does not transfer to test", fontsize=10)
    fig.supxlabel("U-Net + clDice, mean ± std over 3 seeds. Factor fitted on validation, applied to test.", fontsize=7)
    panel(axs[0], "(a)"); panel(axs[1], "(b)"); save("fig4_calibration_null_result")


def fig5_gt_vs_pred_indices(T):
    """T: n x keys x (gt, pred) array from idx_pairs(Pt, Gt, KEYS) in the notebook, U-Net + clDice, seed 0, test set."""
    from scipy.stats import spearmanr
    KEYS = ["density", "mean_width_px", "area_ratio"]
    NAME2 = {"density": "Length density", "mean_width_px": "Mean width (px)", "area_ratio": "Area ratio"}
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for j, (ax, k) in enumerate(zip(axs, KEYS)):
        g, p = T[:, j, 0], T[:, j, 1]
        ax.scatter(g, p, s=9, alpha=.6, color=OI["blue"], edgecolors="none")
        lim = [0, max(g.max(), p.max()) * 1.05]; ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Ground truth"); ax.set_ylabel("Predicted"); ax.set_title(f"{NAME2[k]}\nrho = {spearmanr(g, p)[0]:.2f}, n = {len(g)}", fontsize=9)
        panel(ax, "(" + "abc"[j] + ")")
    plt.tight_layout(); save("fig5_gt_vs_pred_indices")


def fig6_qualitative_error_maps(Pt, Gt, test):
    """Pt, Gt: boolean test-set prediction and ground-truth masks; test: list of (image path, label path)."""
    from PIL import Image
    iou = np.array([(p & g).sum() / max((p | g).sum(), 1) for p, g in zip(Pt, Gt)])
    rng = np.random.RandomState(0)
    sel = list(rng.choice(len(iou), 2, replace=False)) + list(np.argsort(iou)[:2])
    tags = ["random", "random", "worst", "2nd worst"]
    fig, axs = plt.subplots(4, 4, figsize=(8, 5.8))
    for r_, (i, t) in enumerate(zip(sel, tags)):
        img = np.asarray(Image.open(test[i][0]).convert("RGB").resize((544, 384)))
        p, g = Pt[i], Gt[i]
        err = np.zeros((*p.shape, 3)); err[p & g] = [1, 1, 1]; err[p & ~g] = [.84, .37, 0]; err[~p & g] = [0, .45, .7]
        for c_, (im, ttl) in enumerate([(img, "Image"), (g, "Ground truth"), (p, "Prediction"), (err, "Error map")]):
            ax = axs[r_, c_]; ax.imshow(im, cmap="gray"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values(): s.set_visible(False)
            if r_ == 0: ax.set_title(ttl)
        axs[r_, 0].set_ylabel(f"{t}\nIoU {iou[i]:.2f}", fontsize=8)
    fig.legend(handles=[Patch(color="white", ec="k", label="True positive"), Patch(color=(.84, .37, 0), label="False positive"),
                        Patch(color=(0, .45, .7), label="False negative")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.03))
    fig.text(.5, -.07, "Images and annotations: DeepCrack (Liu et al., Neurocomputing 2019). Model: U-Net + clDice, seed 0.", ha="center", fontsize=7)
    save("fig6_qualitative_error_maps")


if __name__ == "__main__":
    fig1_segmentation_comparison()
    fig2_index_fidelity()
    fig3_spur_pruning()
    fig4_calibration_null_result()
    print("Wrote fig1 to fig4 to", FIG, "(fig5 and fig6 need predictions, fig7 was drawn separately)")
