#!/usr/bin/env python3
"""
Real-data validation on the BurstGPT trace (HPMLL/BurstGPT, CC-BY-4.0).

Two modes:

  --real-latency   (recommended; needs a v2.0 file with an 'Elapsed time'
                   column, e.g. BurstGPT_without_fails_3.csv)
                   Computes the quantile-fairness metrics directly on the
                   REAL measured end-to-end latency, split by a real cohort.
                   No simulation, no modelling assumptions.

  (default)        Drives the queue with the real request tokens and compares
                   FCFS vs SJF. Useful to study a scheduler, but the latency is
                   simulated and, on heavy-tailed real tokens at high load, the
                   tail can be dominated by queueing rather than the cohort.

Cohorts (real trace fields): --cohort model  -> GPT-4 vs ChatGPT
                             --cohort logtype -> Conversation vs API

Examples
  python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort model
  python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort logtype
"""

import argparse

import numpy as np

from src.simulation import simulate, poisson_arrivals
from src.metrics import (group_quantiles, parity_gap, quantile_ratio,
                        slo_violation_rates, workload_conditioned_gap, quantile)
from src import trace_adapters


def analyse(resp, group, size, label, SLO, unit=""):
    p50 = group_quantiles(resp, group, 0.50)
    p99 = group_quantiles(resp, group, 0.99)
    v = slo_violation_rates(resp, group, SLO)
    wc = workload_conditioned_gap(resp, group, size, 0.99, "B", "A", n_bins=5)
    gap = parity_gap(resp, group, 0.99, "B", "A")
    print(f"  [{label}]")
    print(f"    p50  A={p50['A']:.3g}{unit}  B={p50['B']:.3g}{unit}")
    print(f"    p99  A={p99['A']:.3g}{unit}  B={p99['B']:.3g}{unit}  "
          f"gap={gap:.3g}{unit}  ratio={quantile_ratio(resp, group, 0.99, 'B', 'A'):.2f}x")
    print(f"    SLO-violation disparity (B-A): {100*(v['B']-v['A']):.1f} pct. points "
          f"(SLO={SLO:.3g}{unit})")
    print(f"    workload-adjusted p99 gap   : {wc['adjusted']:.3g}{unit}  "
          f"(scheduling/other-induced share {100*(1-wc['composition_share']):.0f}%, "
          f"bins {wc['bins_used']}/{wc['bins_total']})")
    return gap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--real-latency", action="store_true",
                    help="analyse the trace's real measured Elapsed time (no sim)")
    ap.add_argument("--cohort", choices=["model", "logtype"], default="model")
    ap.add_argument("--cost", choices=["total", "decode", "weighted"], default="total")
    ap.add_argument("--servers", type=int, default=8)
    ap.add_argument("--target-util", type=float, default=0.85)
    ap.add_argument("--arrivals", choices=["poisson", "trace"], default="poisson")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="leading window size (0 = all rows)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_name = "GPT-4" if args.cohort == "model" else "Conversation"

    # ---------------- real measured latency (recommended) ----------------
    if args.real_latency:
        elapsed, size, group = trace_adapters.load_burstgpt_latency(
            args.csv, cohort=args.cohort, max_requests=(args.max_requests or None))
        b_share = float((group == "B").mean())
        SLO = quantile(elapsed, 0.95)
        print("\nBurstGPT real-data validation  (REAL measured latency, no simulation)")
        print(f"  requests: {len(elapsed):,}   cohort: {args.cohort} "
              f"(B = {a_name}, {100*b_share:.0f}% of traffic)")
        print(f"  mean request size  A={size[group=='A'].mean():.0f}  "
              f"B={size[group=='B'].mean():.0f} tokens  "
              f"(B/A = {size[group=='B'].mean()/size[group=='A'].mean():.2f}x)\n")
        analyse(elapsed, group, size, "observed Elapsed time", SLO, unit="s")
        print("\n  Latency is the value Azure actually measured per request; the "
              "cohort is a real trace field. The workload-adjusted gap conditions "
              "on total tokens, so it separates 'B is slower because its requests "
              "are larger' from any residual difference.")
        print("  Cite the BurstGPT KDD'25 paper; data is CC-BY-4.0.")
        return

    # ---------------- simulated scheduler comparison ----------------
    max_req = args.max_requests or 400000          # cap sim for tractability
    arrival, size, group = trace_adapters.load_burstgpt(
        args.csv, cohort=args.cohort, cost=args.cost, max_requests=max_req)
    b_share = float((group == "B").mean())
    print(f"\nBurstGPT scheduler comparison  (SIMULATED latency)")
    print(f"  requests: {len(size):,}   cohort: {args.cohort} "
          f"(B = {a_name}, {100*b_share:.0f}% of traffic)   cost: {args.cost}")
    print(f"  servers: {args.servers}   arrivals: {args.arrivals}   "
          f"target util: {args.target_util}")
    print(f"  mean request size  A={size[group=='A'].mean():.0f}  "
          f"B={size[group=='B'].mean():.0f} tokens  "
          f"(B/A = {size[group=='B'].mean()/size[group=='A'].mean():.2f}x)")
    print("  NOTE: simulated latency on heavy-tailed real tokens can be queueing-"
          "dominated; prefer --real-latency for publishable numbers.\n")

    if args.arrivals == "trace":
        span = float(arrival.max() - arrival.min()) or 1.0
        lam = len(arrival) / span
        size = size * (args.target_util * args.servers / (lam * size.mean()))
    else:
        arrival = poisson_arrivals(len(size), args.target_util,
                                   np.random.default_rng(args.seed + 7),
                                   args.servers, float(size.mean()))

    warm = int(0.10 * len(size))
    base = simulate(arrival, size, arrival.copy(), args.servers)[warm:]
    SLO = quantile(base, 0.95)
    for label, score in [("FCFS", arrival.copy()), ("SJF", size.copy())]:
        resp = simulate(arrival, size, score, args.servers)[warm:]
        analyse(resp, group[warm:], size[warm:], label, SLO)
        print()


if __name__ == "__main__":
    main()
