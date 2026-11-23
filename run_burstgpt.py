#!/usr/bin/env python3
"""
Real-data validation on the BurstGPT trace (HPMLL/BurstGPT, CC-BY-4.0).

Two modes:

  --real-latency   (recommended; needs a v2.0 file with an 'Elapsed time'
                   column, e.g. BurstGPT_without_fails_3.csv)
                   Computes the quantile-fairness metrics directly on the REAL
                   measured end-to-end latency, split by a real cohort, with
                   bootstrap 95% confidence intervals, SLO-attainment disparity
                   across SLO strictness, and the workload-conditioned
                   decomposition under two cost proxies (total vs response
                   tokens). No simulation, no modelling assumptions.

  (default)        Drives the queue with the real request tokens and compares
                   FCFS vs SJF (simulated latency; see the note it prints).

Cohorts (real trace fields): --cohort model  -> GPT-4 vs ChatGPT
                             --cohort logtype -> Conversation vs API

Examples
  python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort model --n-boot 200
  python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort logtype --n-boot 200
"""

import argparse

import numpy as np

from src.simulation import simulate, poisson_arrivals
from src.metrics import (group_quantiles, parity_gap, quantile_ratio,
                        slo_violation_rates, workload_conditioned_gap,
                        quantile, bootstrap_ci)
from src import trace_adapters


def real_latency_report(args):
    elapsed, total, request, response, group = trace_adapters.load_burstgpt_latency(
        args.csv, cohort=args.cohort, max_requests=(args.max_requests or None))
    n = len(elapsed)
    isA, isB = group == "A", group == "B"
    a_name = "GPT-4" if args.cohort == "model" else "Conversation"
    SLO = quantile(elapsed, 0.95)

    # --- point estimates ---
    qa = quantile(elapsed[isA], 0.99)
    qb = quantile(elapsed[isB], 0.99)
    gap = qb - qa
    ratio = qb / qa
    wc_total = workload_conditioned_gap(elapsed, group, total, 0.99, "B", "A", n_bins=5)
    wc_resp = workload_conditioned_gap(elapsed, group, response, 0.99, "B", "A", n_bins=5)

    print("\nBurstGPT real-data validation  (REAL measured latency, no simulation)")
    print(f"  requests: {n:,}   cohort: {args.cohort} "
          f"(B = {a_name}, {100*isB.mean():.0f}% of traffic)")
    print(f"  mean request size  A={total[isA].mean():.0f}  B={total[isB].mean():.0f} "
          f"tokens  (B/A = {total[isB].mean()/total[isA].mean():.2f}x)")

    # --- bootstrap CIs ---
    if args.n_boot > 0:
        def g_stat(idx):
            e, g = elapsed[idx], group[idx]
            return quantile(e[g == "B"], 0.99) - quantile(e[g == "A"], 0.99)

        def r_stat(idx):
            e, g = elapsed[idx], group[idx]
            return quantile(e[g == "B"], 0.99) / quantile(e[g == "A"], 0.99)

        def adj_stat(idx):
            return workload_conditioned_gap(elapsed[idx], group[idx], total[idx],
                                            0.99, "B", "A", n_bins=5)["adjusted"]

        def disp_stat(idx):
            e, g = elapsed[idx], group[idx]
            return ((e[g == "B"] > SLO).mean() - (e[g == "A"] > SLO).mean())

        gci = bootstrap_ci(g_stat, n, args.n_boot, seed=args.seed)
        rci = bootstrap_ci(r_stat, n, args.n_boot, seed=args.seed)
        aci = bootstrap_ci(adj_stat, n, args.n_boot, seed=args.seed)
        dci = bootstrap_ci(disp_stat, n, args.n_boot, seed=args.seed)
        ci = lambda t: f"[{t[0]:.2f}, {t[1]:.2f}]"
        boot = f"  (95% CI from {args.n_boot} bootstraps)"
    else:
        gci = rci = aci = dci = None
        ci = lambda t: ""
        boot = "  (run with --n-boot 200 for 95% CIs)"

    print(f"\n  [observed Elapsed time, p99]{boot}")
    print(f"    p99 A = {qa:.1f} s    p99 B = {qb:.1f} s")
    print(f"    p99 parity gap (B-A) = {gap:.2f} s   {ci(gci) if gci else ''}")
    print(f"    p99 ratio (B/A)      = {ratio:.3f}x  {ci(rci) if rci else ''}")

    print(f"\n  SLO-violation disparity (B-A) across SLO strictness:")
    for p, label in [(0.90, "p90"), (0.95, "p95"), (0.99, "p99")]:
        T = quantile(elapsed, p)
        d = (elapsed[isB] > T).mean() - (elapsed[isA] > T).mean()
        extra = f"   {ci(dci)}" if (p == 0.95 and dci) else ""
        print(f"    SLO = overall {label} ({T:.1f} s): {100*d:+.1f} pct. points{extra}")

    print(f"\n  Workload-conditioned p99 gap (scheduling/other-induced residual):")
    print(f"    conditioned on TOTAL tokens   : {wc_total['adjusted']:.2f} s "
          f"{ci(aci) if aci else ''}  "
          f"(composition explains {100*(1-(1 if wc_total['raw']==0 else wc_total['adjusted']/wc_total['raw'])):.0f}% of a {wc_total['raw']:.1f} s raw gap)")
    print(f"    conditioned on RESPONSE tokens: {wc_resp['adjusted']:.2f} s  "
          f"(isolates per-token/decode effects)")

    print("\n  Latency is the value Azure actually measured; the cohort is a real "
          "trace field. A near-zero or negative adjusted gap means the raw gap is "
          "explained by workload composition, not per-request scheduling bias.")
    print("  Cite the BurstGPT KDD'25 paper; data is CC-BY-4.0.")


def sim_compare(args):
    arrival, size, group = trace_adapters.load_burstgpt(
        args.csv, cohort=args.cohort, cost="total",
        max_requests=(args.max_requests or 400000))
    print("\nBurstGPT scheduler comparison (SIMULATED latency)")
    print("  NOTE: simulated latency on heavy-tailed real tokens can be queueing-"
          "dominated; prefer --real-latency for publishable numbers.")
    arrival = poisson_arrivals(len(size), args.target_util,
                               np.random.default_rng(args.seed + 7),
                               args.servers, float(size.mean()))
    warm = int(0.10 * len(size))
    base = simulate(arrival, size, arrival.copy(), args.servers)[warm:]
    SLO = quantile(base, 0.95)
    for label, score in [("FCFS", arrival.copy()), ("SJF", size.copy())]:
        resp = simulate(arrival, size, score, args.servers)[warm:]
        g, z = group[warm:], size[warm:]
        print(f"  [{label}] p99 gap (B-A) = {parity_gap(resp, g, 0.99, 'B', 'A'):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--real-latency", action="store_true")
    ap.add_argument("--cohort", choices=["model", "logtype"], default="model")
    ap.add_argument("--n-boot", type=int, default=0,
                    help="bootstrap resamples for 95%% CIs (0 = off; 200 suggested)")
    ap.add_argument("--servers", type=int, default=8)
    ap.add_argument("--target-util", type=float, default=0.85)
    ap.add_argument("--max-requests", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.real_latency:
        real_latency_report(args)
    else:
        sim_compare(args)


if __name__ == "__main__":
    main()
