# Tail Latency as an Equity Problem

Quantile fairness and quality-of-service disparities in serving systems.

This repository reproduces the experiments in the manuscript *"Tail Latency as
an Equity Problem: Quantile Fairness and Quality-of-Service Disparities in
Serving Systems."* It shows how a healthy-looking aggregate tail latency (p99)
can hide large, scheduling-induced differences in service quality across user
groups — and how to measure and reduce them.

It runs out of the box on **synthetic data**, and the same engine runs on
**real, openly-licensed public traces**.

## What's inside

```
tail-latency-equity/
├── run_synthetic.py        # end-to-end synthetic experiment (no download needed)
├── run_trace.py            # same metrics on a real public trace
├── src/
│   ├── simulation.py       # discrete-event multi-server priority queue + workload
│   ├── metrics.py          # quantile-fairness metrics (parity gap, SLO disparity,
│   │                       #   workload-conditioned decomposition)
│   ├── plots.py            # the three publication figures (300 DPI, colorblind-safe)
│   └── trace_adapters.py   # loaders for Azure / Google / Alibaba public traces
├── scripts/download_data.sh
├── docs/DATA.md            # datasets, licenses, schemas, cohort rules
├── docs/PUBLISHING.md      # novelty + acceptance guidance
├── data/                   # downloaded traces go here (git-ignored)
└── results/                # metrics.json + figs/ (generated)
```

## Quick start (macOS)

```bash
# 1. clone / unzip this repo, then from its folder:
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the synthetic experiment
python run_synthetic.py
```

This writes `results/metrics.json` and three figures to `results/figs/`. It is
fully seeded, so your numbers match the paper exactly (e.g. at 95% load Group B's
p99 ≈ 24.7 vs an aggregate of ≈ 15.6; ≈ 60% of the 85%-load gap is
scheduling-induced; the controller cuts the gap ≈ 72% for < 9% efficiency cost).

### In VS Code
Open the folder (`File → Open Folder`), install the **Python** extension, pick
the `.venv` interpreter (`Cmd+Shift+P → Python: Select Interpreter`), then press
▶ on `run_synthetic.py`.

## Running on real public data

```bash
bash scripts/download_data.sh    # clones dataset repos and prints download links
# download a CSV into data/ (see docs/DATA.md), then:
python run_trace.py --dataset azure-functions --csv data/AzureFunctionsInvocationTrace2021.csv
python run_trace.py --dataset azure-llm       --csv data/your_llm_trace.csv --w 1.0
```

See **[docs/DATA.md](docs/DATA.md)** for datasets and licenses (Azure Public
Dataset is CC-BY-4.0), and **[docs/PUBLISHING.md](docs/PUBLISHING.md)** for how
to frame the work for publication.

## Ethics note

All cohorts in the synthetic experiment are **simulated** ("Group A" / "Group
B"); they are not real people or protected attributes. On real traces, `group`
must be a lawful, consented attribute or a clearly-declared **operational**
cohort (tenant, application, region, request shape) — never a proxy for a
protected class.

## Requirements

Python 3.9+ and the packages in `requirements.txt` (numpy, scipy, matplotlib,
seaborn, pandas).

## License

Code: MIT (see `LICENSE`). Downloaded datasets keep their own licenses.
