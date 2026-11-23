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
    Cohort: long-context vs short-context requests (a neutral request-shape
    split that is useful for studying TTFT-style tail fairness).
    """
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["TIMESTAMP"])
    arrival = (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
    ctx = df["ContextTokens"].to_numpy(dtype=float)
    gen = df["GeneratedTokens"].to_numpy(dtype=float)
    size = ctx + gen
    group = np.where(ctx > np.median(ctx), "B", "A")   # long-context cohort = B
    order = np.argsort(arrival, kind="mergesort")
    return arrival[order], size[order], group[order]
