#!/usr/bin/env python3
"""
Run the SAME experiment on a real, openly-licensed public trace.

Examples
  python run_trace.py --dataset azure-functions --csv data/AzureFunctionsInvocationTrace2021.csv
  python run_trace.py --dataset azure-llm       --csv data/AzureLLMInferenceTrace_conv.csv --w 1.0

The trace supplies real arrival times and service demands. The cohort label is
a neutral OPERATIONAL split (see src/trace_adapters.py) — not a demographic
attribute. The latency-tolerance proxy is an explicit experimental factor,
clearly synthetic, since no real trace carries one.
"""

import argparse

import numpy as np

from src.simulation import simulate, tolerance_score
from src.metrics import (group_quantiles, parity_gap, slo_violation_rates,
                        workload_conditioned_gap, quantile)
from src import trace_adapters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["azure-functions", "azure-llm"], required=True)
    ap.add_argument("--csv", required=True, help="path to the downloaded trace CSV")
    ap.add_argument("--servers", type=int, default=8)
    ap.add_argument("--target-util", type=float, default=0.85,
                    help="service times are rescaled to hit this utilisation")
    ap.add_argument("--w", type=float, default=0.0, help="fairness-controller weight")
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--tol-delta", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "azure-functions":
        arrival, size, group = trace_adapters.load_azure_functions(args.csv)
    else:
        arrival, size, group = trace_adapters.load_azure_llm(args.csv)

    # Rescale service demands so offered load hits the target utilisation.
    span = float(arrival.max() - arrival.min()) or 1.0
    lam = len(arrival) / span
    scale = args.target_util * args.servers / (lam * size.mean())
    size = size * scale

    rng = np.random.default_rng(args.seed)
    tol = rng.normal(0.0, 1.0, len(group)) + np.where(group == "B", args.tol_delta, 0.0)
    pred = size * rng.lognormal(0.0, 0.30, len(size))
    score = tolerance_score(pred, tol, group, args.w, args.beta)

    resp = simulate(arrival, size, score, args.servers)
    warm = int(0.10 * len(resp))
    resp, group, size = resp[warm:], group[warm:], size[warm:]

    SLO = quantile(resp, 0.95)
    q = group_quantiles(resp, group, 0.99)
    v = slo_violation_rates(resp, group, SLO)
    wc = workload_conditioned_gap(resp, group, size, 0.99, "B", "A", n_bins=5)

    print(f"\nDataset: {args.dataset}   requests analysed: {len(resp):,}   "
          f"servers: {args.servers}   target util: {args.target_util}   w: {args.w}\n")
    print(f"  aggregate p99      : {quantile(resp, 0.99):.3f}")
    print(f"  Group A p99        : {q['A']:.3f}")
    print(f"  Group B p99        : {q['B']:.3f}")
    print(f"  p99 parity gap     : {parity_gap(resp, group, 0.99, 'B', 'A'):.3f}")
    print(f"  SLO (overall p95)  : {SLO:.3f}")
    print(f"  SLO-viol disparity : {100*(v['B']-v['A']):.1f} pct. points")
    print(f"  raw gap            : {wc['raw']:.3f}")
    print(f"  workload-adjusted  : {wc['adjusted']:.3f}  "
          f"(scheduling-induced share: {100*(1-wc['composition_share']):.0f}%)")
    print("\nNote: 'group' is a neutral operational cohort, not a demographic "
          "attribute. Document your cohort choice in any publication.")


if __name__ == "__main__":
    main()
