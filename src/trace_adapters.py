"""
Load openly-licensed PUBLIC traces into the (arrival, size, group) arrays the
simulation consumes.

READ THIS FIRST — cohort labels and ethics
------------------------------------------
Real production traces contain NO protected attributes. The "group" returned
here must be a lawful, consented attribute or a clearly-declared OPERATIONAL
cohort (tenant, application, region, request shape, priority) — never a proxy
for a protected class. The helpers below derive a cohort from a neutral
operational key solely to demonstrate the metrics on real workloads. State your
cohort definition explicitly in any paper, and never present an operational
split as if it were a demographic one.

Schemas below match the datasets at the time of writing; always confirm the
column names against each dataset's own README/.md before running.
"""

import numpy as np
import pandas as pd


def load_azure_functions(csv_path, group_key="app", n_groups=2):
    """Azure Functions Invocation Trace 2021 (Azure Public Dataset, CC-BY-4.0).

    Expected columns: app, func, end_timestamp, duration.
    Cohort: requests are split by a neutral operational key (default: the
    application id, hashed into n_groups buckets).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["end_timestamp", "duration"]).sort_values("end_timestamp")
    size = df["duration"].to_numpy(dtype=float)
    arrival = (df["end_timestamp"] - df["duration"]).to_numpy(dtype=float)
    arrival = arrival - arrival.min()
    codes = pd.factorize(df[group_key])[0]
    group = np.where(codes % n_groups == 0, "A", "B")
    order = np.argsort(arrival, kind="mergesort")
    return arrival[order], size[order], group[order]


def load_azure_llm(csv_path):
    """Azure LLM Inference Dataset 2023 (Azure Public Dataset, CC-BY-4.0).

    Expected columns: TIMESTAMP, ContextTokens, GeneratedTokens.
    Service demand is approximated by total tokens (context + generated).
    Cohort: decode-heavy vs prompt-heavy requests (group B = more generated
    tokens than the median). This is a neutral request-shape split that is
    useful for TTFT-style tail fairness, and it deliberately overlaps with
    total size (a request can be large-context/short-output or the reverse),
    so the workload-conditioned decomposition has within-bin overlap to work
    with. Swap in your own cohort by editing this line.
    """
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["TIMESTAMP"])
    arrival = (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
    ctx = df["ContextTokens"].to_numpy(dtype=float)
    gen = df["GeneratedTokens"].to_numpy(dtype=float)
    size = ctx + gen
    group = np.where(gen > np.median(gen), "B", "A")   # decode-heavy cohort = B
    order = np.argsort(arrival, kind="mergesort")
    return arrival[order], size[order], group[order]


def load_burstgpt(csv_path, cohort="model", cost="total", max_requests=None):
    """BurstGPT (HPMLL/BurstGPT, CC-BY-4.0): a real Azure OpenAI serving trace.

    v1-format columns: Timestamp, Model, Request tokens, Response tokens,
    Total tokens, Log Type. (Newer v2.0 files add Session ID and Elapsed time.)

    cohort : "model"   -> B = GPT-4, A = ChatGPT  (a real model-tier cohort)
             "logtype" -> B = Conversation log, A = API log
    cost   : service-demand proxy from real tokens:
             "total"    -> request + response tokens
             "decode"   -> response tokens only (decode dominates LLM latency)
             "weighted" -> 0.2*request + 1.0*response (cheap prefill, costly decode)
    max_requests : take a contiguous leading window (for a stable, quick run).

    Returns (arrival, size, group), sorted by arrival. Failed/empty requests
    (Total tokens <= 0) are dropped. The cohort is a REAL field in the trace,
    not a synthetic or inferred attribute.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Total tokens"] > 0]
    if max_requests:
        df = df.iloc[:int(max_requests)]
    req = df["Request tokens"].to_numpy(dtype=float)
    res = df["Response tokens"].to_numpy(dtype=float)
    if cost == "decode":
        size = res
    elif cost == "weighted":
        size = 0.2 * req + 1.0 * res
    else:
        size = req + res
    arrival = df["Timestamp"].to_numpy(dtype=float)
    arrival = arrival - arrival.min()
    if cohort == "logtype":
        group = np.where(df["Log Type"].to_numpy() == "Conversation log", "B", "A")
    else:
        group = np.where(df["Model"].to_numpy() == "GPT-4", "B", "A")
    order = np.argsort(arrival, kind="mergesort")
    return arrival[order], size[order], group[order]


def load_burstgpt_latency(csv_path, cohort="model", size_col="total", max_requests=None):
    """Load BurstGPT with its REAL measured end-to-end latency.

    Requires a v2.0-format file (e.g. BurstGPT_without_fails_3.csv) that has an
    'Elapsed time' column. Returns (elapsed, size, group), where elapsed is the
    observed latency in seconds, size is the token-based workload used for the
    workload-conditioned decomposition, and group is a real trace cohort.

    This is the simulation-free path: the fairness metrics are computed on
    latencies actually observed in Azure OpenAI production serving.
    """
    df = pd.read_csv(csv_path)
    if "Elapsed time" not in df.columns:
        raise ValueError(
            "This file has no 'Elapsed time' column. Use a BurstGPT v2.0 file "
            "such as BurstGPT_without_fails_3.csv.")
    df = df[(df["Total tokens"] > 0) & (df["Elapsed time"] > 0)]
    if max_requests:
        df = df.iloc[:int(max_requests)]
    elapsed = df["Elapsed time"].to_numpy(dtype=float)
    if size_col == "response":
        size = df["Response tokens"].to_numpy(dtype=float)
    else:
        size = df["Total tokens"].to_numpy(dtype=float)
    if cohort == "logtype":
        group = np.where(df["Log Type"].to_numpy() == "Conversation log", "B", "A")
    else:
        group = np.where(df["Model"].to_numpy() == "GPT-4", "B", "A")
    return elapsed, size, group
