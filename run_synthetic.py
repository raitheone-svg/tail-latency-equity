#!/usr/bin/env python3
"""
Run the full SYNTHETIC experiment end to end:

    python run_synthetic.py

Writes results/metrics.json and the three figures into results/figs/.
Everything is seeded, so results are identical on every machine.

Experiments
  1. Per-group vs aggregate p99 across system load (proxy scheduler).
  2. Workload-conditioned decomposition of the p99 gap at 85% load.
  3. Efficiency-fairness trade-off as the controller weight w varies.
"""

import json
import pathlib

import numpy as np

from src.simulation import (simulate, synthetic_workload,
                            tolerance_score, poisson_arrivals)
from src.metrics import (group_quantiles, parity_gap, slo_violation_rates,
                        workload_conditioned_gap, quantile)
from src import plots

# --- configuration (matches the manuscript) ---------------------------------
ROOT = pathlib.Path(__file__).parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"

MASTER_SEED = 20251122
C_SERVERS = 8
N = 60_000
SEEDS = [0, 1, 2, 3]
WARMUP = 0.10

P_B = 0.40
MEAN_A, SIG_A = 0.80, 0.50
MEAN_B, SIG_B = 1.40, 0.90
E_S = (1 - P_B) * MEAN_A + P_B * MEAN_B      # 1.04
BETA = 1.50
TOL_DELTA = 1.20


def run(rho, w, seeds=SEEDS):
    """Pool several seeded runs; return post-warm-up (resp, group, size)."""
    R, G, Z = [], [], []
    for s in seeds:
        rng = np.random.default_rng(MASTER_SEED + s)
        group, size, pred, tol = synthetic_workload(
            N, rng, P_B, MEAN_A, SIG_A, MEAN_B, SIG_B, 0.30, TOL_DELTA)
        arrival = poisson_arrivals(N, rho, rng, C_SERVERS, E_S)   # already sorted
        score = tolerance_score(pred, tol, group, w, BETA)
        resp = simulate(arrival, size, score, C_SERVERS)
        warm = int(WARMUP * N)
        R.append(resp[warm:]); G.append(group[warm:]); Z.append(size[warm:])
    return np.concatenate(R), np.concatenate(G), np.concatenate(Z)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # --- Experiment 1: load sweep (proxy scheduler, w = 0) ---
    loads = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    exp1 = {"load": [], "agg_p99": [], "A_p99": [], "B_p99": []}
    for rho in loads:
        resp, group, _ = run(rho, 0.0)
        q = group_quantiles(resp, group, 0.99)
        exp1["load"].append(rho)
        exp1["agg_p99"].append(quantile(resp, 0.99))
        exp1["A_p99"].append(q["A"])
        exp1["B_p99"].append(q["B"])

    # --- Experiment 2: workload-conditioned decomposition (rho = 0.85) ---
    resp, group, size = run(0.85, 0.0)
    wc = workload_conditioned_gap(resp, group, size, 0.99, "B", "A", n_bins=3)
    exp2 = {
        "bins": ["Small", "Medium", "Large"],
        "A_p99_bin": [b["lo"] for b in wc["bins"]],
        "B_p99_bin": [b["hi"] for b in wc["bins"]],
        "raw_gap": wc["raw"], "adj_gap": wc["adjusted"],
        "composition_share": wc["composition_share"],
    }

    # --- Experiment 3: efficiency-fairness trade-off (rho = 0.85) ---
    base_resp, base_group, _ = run(0.85, 0.0)
    SLO = quantile(base_resp, 0.95)
    ws = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    exp3 = {"w": [], "agg_p99": [], "gap_p99": [], "slo_disp": [], "SLO": SLO}
    for w in ws:
        resp, group, _ = run(0.85, w)
        v = slo_violation_rates(resp, group, SLO)
        exp3["w"].append(w)
        exp3["agg_p99"].append(quantile(resp, 0.99))
        exp3["gap_p99"].append(parity_gap(resp, group, 0.99, "B", "A"))
        exp3["slo_disp"].append(v["B"] - v["A"])

    metrics = {
        "config": {"C_SERVERS": C_SERVERS, "P_B": P_B, "E_S": E_S,
                   "N_per_run": N, "seeds": SEEDS, "master_seed": MASTER_SEED},
        "exp1_load_sweep": exp1,
        "exp2_workload_conditioned": exp2,
        "exp3_tradeoff": exp3,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    figs = plots.generate(metrics, FIGS)

    # --- console summary ---
    print("Synthetic experiment complete.\n")
    print("Exp 1 - p99 across load (w=0):")
    for i, rho in enumerate(exp1["load"]):
        print(f"  rho={rho:.2f}  agg={exp1['agg_p99'][i]:6.2f}  "
              f"A={exp1['A_p99'][i]:6.2f}  B={exp1['B_p99'][i]:6.2f}  "
              f"gap={exp1['B_p99'][i]-exp1['A_p99'][i]:6.2f}")
    print(f"\nExp 2 - workload-conditioned (rho=0.85): "
          f"raw gap={exp2['raw_gap']:.2f}, adjusted={exp2['adj_gap']:.2f}, "
          f"scheduling-induced={100*(1-exp2['composition_share']):.0f}%")
    print(f"\nExp 3 - trade-off (rho=0.85, SLO={SLO:.2f}):")
    for i, w in enumerate(exp3["w"]):
        print(f"  w={w:.2f}  agg_p99={exp3['agg_p99'][i]:6.2f}  "
              f"gap={exp3['gap_p99'][i]:6.2f}  "
              f"SLO_disp={exp3['slo_disp'][i]*100:5.1f}pp")
    print("\nWrote: results/metrics.json")
    for f in figs:
        print("Wrote:", f)


if __name__ == "__main__":
    main()
