"""
Publication-ready figures (colorblind-safe Okabe-Ito palette, 300 DPI).
`generate(metrics, outdir)` writes the three manuscript figures from the
dict produced by run_synthetic.py.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OK = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "black": "#000000", "gray": "#5A5A5A",
      "purple": "#CC79A7"}


def _rc():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def fig_load_sweep(e1, path):
    x = np.array(e1["load"]) * 100
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(x, e1["B_p99"], "-o", color=OK["vermillion"], lw=2.2, ms=6, label="Group B  p99")
    ax.plot(x, e1["agg_p99"], "--s", color=OK["black"], lw=2.0, ms=5, label="Aggregate p99 (all users)")
    ax.plot(x, e1["A_p99"], "-^", color=OK["blue"], lw=2.2, ms=6, label="Group A  p99")
    ax.fill_between(x, e1["A_p99"], e1["B_p99"], color=OK["vermillion"], alpha=0.08)
    ax.set_xlabel("System load (% utilisation)")
    ax.set_ylabel("99th-percentile response time (time units)")
    ax.set_title("Aggregate p99 masks a widening group gap at the tail")
    ax.set_ylim(0, max(e1["B_p99"]) * 1.12)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_workload_conditioned(e2, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    bins = e2["bins"]
    xb = np.arange(len(bins)); w = 0.38
    axL.bar(xb - w/2, e2["A_p99_bin"], w, color=OK["blue"], label="Group A")
    axL.bar(xb + w/2, e2["B_p99_bin"], w, color=OK["vermillion"], label="Group B")
    axL.set_xticks(xb); axL.set_xticklabels([f"{b}\njobs" for b in bins])
    axL.set_ylabel("Within-bin p99 (time units)")
    axL.set_xlabel("Workload bin (service-demand tercile)")
    axL.set_title("Per-group p99 within matched workload bins"); axL.legend(loc="upper left")

    axR.bar([0, 1], [e2["raw_gap"], e2["adj_gap"]], 0.6, color=[OK["gray"], OK["green"]])
    axR.set_xticks([0, 1]); axR.set_xticklabels(["Raw\np99 gap", "Workload-\nadjusted gap"])
    axR.set_ylabel("Cross-group p99 gap (time units)")
    axR.set_title("Decomposing the gap")
    for i, v in enumerate([e2["raw_gap"], e2["adj_gap"]]):
        axR.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_tradeoff(e3, path):
    gap = np.array(e3["gap_p99"]); agg = np.array(e3["agg_p99"])
    ws = np.array(e3["w"]); disp = np.array(e3["slo_disp"]) * 100
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.6))
    axL.plot(gap, agg, "-o", color=OK["purple"], lw=2.2, ms=7)
    for g, a, wv in zip(gap, agg, ws):
        axL.annotate(f"w={wv:g}", (g, a), textcoords="offset points",
                     xytext=(4, 5), fontsize=8.5, color=OK["gray"])
    axL.invert_xaxis()
    axL.set_xlabel("Cross-group p99 gap  (lower = fairer)")
    axL.set_ylabel("Aggregate p99  (lower = more efficient)")
    axL.set_title("Efficiency-fairness frontier")

    axR.plot(ws, disp, "-D", color=OK["orange"], lw=2.2, ms=6)
    axR.axhline(0, color=OK["gray"], lw=0.8, ls=":")
    axR.set_xlabel("Fairness-controller weight  w")
    axR.set_ylabel("SLO-violation disparity (B - A, pct. points)")
    axR.set_title("Closing the SLO-violation gap")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def generate(metrics, outdir):
    _rc()
    import pathlib
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig_load_sweep(metrics["exp1_load_sweep"], outdir / "fig1_load_p99.png")
    fig_workload_conditioned(metrics["exp2_workload_conditioned"], outdir / "fig2_workload_conditioned.png")
    fig_tradeoff(metrics["exp3_tradeoff"], outdir / "fig3_tradeoff.png")
    return sorted(str(p) for p in outdir.glob("*.png"))
