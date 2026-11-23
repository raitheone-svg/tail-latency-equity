"""
Discrete-event multi-server priority queue, plus the synthetic workload and
scheduler used in the paper.

The core `simulate()` is data-agnostic: give it arrival times, service demands,
and a per-request priority score, and it returns response times. That lets the
exact same engine run on synthetic data and on real public traces.
"""

import heapq
import numpy as np


def simulate(arrival, size, score, c_servers=8):
    """Non-preemptive priority queue on c_servers identical servers.

    arrival : arrival times (any order; sorted internally)
    size    : service demand per request
    score   : priority score per request; LOWER is served first
    returns : response times (completion - arrival), aligned with the inputs
    """
    arrival = np.asarray(arrival, dtype=float)
    size = np.asarray(size, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(arrival)
    order = np.argsort(arrival, kind="mergesort")

    response = np.empty(n)
    free = c_servers
    completions = []   # heap of (completion_time, idx)
    waiting = []       # heap of (score, arrival_time, idx)
    i = 0

    while i < n or completions:
        next_arr = arrival[order[i]] if i < n else np.inf
        next_comp = completions[0][0] if completions else np.inf

        if next_arr <= next_comp:
            j = order[i]
            i += 1
            t = arrival[j]
            if free > 0:
                free -= 1
                heapq.heappush(completions, (t + size[j], j))
                response[j] = size[j]                 # waited 0
            else:
                heapq.heappush(waiting, (score[j], t, j))
        else:
            t, _ = heapq.heappop(completions)
            free += 1
            if waiting:
                _, a, j = heapq.heappop(waiting)
                free -= 1
                heapq.heappush(completions, (t + size[j], j))
                response[j] = (t + size[j]) - a

    return response


def synthetic_workload(n, rng, p_b=0.40, mean_a=0.80, sig_a=0.50,
                       mean_b=1.40, sig_b=0.90, pred_sig=0.30, tol_delta=1.20):
    """Draw simulated cohorts, service demands, a noisy size predictor, and a
    group-correlated latency-tolerance proxy.

    Group B (the minority cohort) has a heavier, more variable workload, and is
    inferred as more latency-tolerant on average. These are SIMULATED cohorts,
    not real attributes.
    """
    def mu(mean, sig):
        return np.log(mean) - 0.5 * sig ** 2

    is_b = rng.random(n) < p_b
    size = np.where(is_b,
                    rng.lognormal(mu(mean_b, sig_b), sig_b, n),
                    rng.lognormal(mu(mean_a, sig_a), sig_a, n))
    pred = size * rng.lognormal(0.0, pred_sig, n)          # ~30% prediction noise
    tol = rng.normal(0.0, 1.0, n) + np.where(is_b, tol_delta, 0.0)
    group = np.where(is_b, "B", "A")
    return group, size, pred, tol


def tolerance_score(pred, tol, group, w, beta=1.5):
    """Scheduler priority score (lower served first):

        score = pred_size + beta * (tolerance - w * group_mean_tolerance)

    w = 0   : pure proxy scheduler (defers the 'tolerant' cohort)
    w = 1   : re-centres each group's tolerance on its own mean (removes the
              systematic group deferral, keeps within-group ordering)
    w > 1   : active compensation for the deferred cohort
    """
    centers = {g: tol[group == g].mean() for g in np.unique(group)}
    center = np.array([centers[g] for g in group])
    return pred + beta * (tol - w * center)


def poisson_arrivals(n, rho, rng, c_servers=8, mean_demand=1.04):
    """Poisson arrivals whose rate targets utilisation rho on c_servers."""
    lam = rho * c_servers / mean_demand
    return np.cumsum(rng.exponential(1.0 / lam, n))
