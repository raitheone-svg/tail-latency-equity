"""
Quantile-fairness metrics (Section 3 of the manuscript).

Every metric is a small, reusable function over three aligned arrays:
    resp   : response (latency) times, one per request
    group  : cohort label per request (e.g. "A" / "B")
    size   : service demand per request (for workload conditioning)

No metric assumes anything about where the data came from, so the same
functions apply to the synthetic simulation and to real public traces.
"""

import numpy as np


def quantile(x, p):
    """The p-quantile of array x (p in (0, 1); p=0.99 is the p99 tail)."""
    return float(np.quantile(x, p))


def group_quantiles(resp, group, p):
    """p-quantile per group, as a dict {group_label: value}."""
    return {g: quantile(resp[group == g], p) for g in np.unique(group)}


def parity_gap(resp, group, p, g_hi, g_lo):
    """Cross-group quantile parity gap: Q_{g_hi}(p) - Q_{g_lo}(p).

    Positive means g_hi is slower than g_lo at quantile p.
    """
    return quantile(resp[group == g_hi], p) - quantile(resp[group == g_lo], p)


def quantile_ratio(resp, group, p, g_hi, g_lo):
    """Ratio form of the parity gap: Q_{g_hi}(p) / Q_{g_lo}(p)."""
    return quantile(resp[group == g_hi], p) / quantile(resp[group == g_lo], p)


def max_parity_gap(resp, group, p):
    """Worst-case spread at quantile p: max group quantile minus min group quantile."""
    qs = group_quantiles(resp, group, p)
    return max(qs.values()) - min(qs.values())


def slo_violation_rates(resp, group, T):
    """Fraction of each group's requests slower than the SLO target T."""
    return {g: float((resp[group == g] > T).mean()) for g in np.unique(group)}


def slo_violation_disparity(resp, group, T, g_hi, g_lo):
    """SLO-violation disparity: V_{g_hi}(T) - V_{g_lo}(T), in [0, 1]."""
    r = slo_violation_rates(resp, group, T)
    return r[g_hi] - r[g_lo]


def workload_conditioned_gap(resp, group, size, p, g_hi, g_lo, n_bins=3):
    """Direct-standardization decomposition of the p-quantile parity gap.

    Splits requests into n_bins service-demand bins, computes each group's
    p-quantile within each bin, and recombines using a single shared weight
    per bin (the overall share of requests in that bin). The result separates:

        raw          : Q_{g_hi}(p) - Q_{g_lo}(p) over all requests
        adjusted     : the same gap holding workload mix constant
                       (the SCHEDULING-INDUCED component)
        composition  : raw - adjusted (the WORKLOAD component)

    Use finer bins when within-bin work must be more comparable.
    """
    edges = np.quantile(size, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.inf
    idx = np.digitize(size, edges[1:-1])

    adjusted = 0.0
    bins = []
    for b in range(n_bins):
        m = idx == b
        w = float(m.mean())
        q_hi = quantile(resp[m & (group == g_hi)], p)
        q_lo = quantile(resp[m & (group == g_lo)], p)
        adjusted += w * (q_hi - q_lo)
        bins.append({"bin": int(b), "weight": w,
                     "gap": q_hi - q_lo, "hi": q_hi, "lo": q_lo})

    raw = parity_gap(resp, group, p, g_hi, g_lo)
    return {
        "raw": raw,
        "adjusted": adjusted,
        "composition": raw - adjusted,
        "composition_share": (raw - adjusted) / raw if raw else float("nan"),
        "bins": bins,
    }
