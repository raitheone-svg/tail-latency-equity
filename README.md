# Tail Latency as an Equity Problem

Quantile fairness and quality-of-service disparities in serving systems.

Reproducible code for the paper *"Tail Latency as an Equity Problem: Quantile
Fairness and Quality-of-Service Disparities in Serving Systems"*
(Akhilesh Rai, Sarthak Saptarshi Bandopadhyay, Manash Chanda).

It shows how a healthy-looking aggregate tail latency (p99) can hide
scheduling-induced service-quality gaps across user groups, defines
quantile-fairness metrics to measure them, and validates those metrics on both
a controlled simulation and a real, openly-licensed production trace.

## What's inside

```
tail-latency-equity/
├── run_synthetic.py     # controlled experiment (no download needed)
├── run_trace.py         # simulation driven by a public trace
├── run_burstgpt.py      # real-data validation on BurstGPT (incl. --real-latency)
├── src/
│   ├── simulation.py    # discrete-event multi-server priority queue + workload
│   ├── metrics.py       # quantile-fairness metrics + workload decomposition
│   ├── plots.py         # the three publication figures (300 DPI, colorblind-safe)
│   └── trace_adapters.py# loaders: Azure Functions, Azure LLM, BurstGPT
├── scripts/download_data.sh
├── docs/DATA.md         # datasets, licenses, schemas, cohort rules
└── docs/PUBLISHING.md   # novelty + acceptance guidance
```

## Quick start (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_synthetic.py          # writes results/metrics.json + results/figs/
```

Fully seeded, so the numbers match the paper (at 95% load Group B's p99 ≈ 24.7
vs an aggregate of ≈ 15.6; ≈ 60% of the 85%-load gap is scheduling-induced; the
bias-aware controller cuts the gap ≈ 72% for < 9% efficiency cost).

## Real-data validation (BurstGPT)

[BurstGPT](https://github.com/HPMLL/BurstGPT) is a CC-BY-4.0 trace of ~Azure
OpenAI serving. The v2.0 file `BurstGPT_without_fails_3.csv` records the **real
measured latency** per request plus two real cohorts, so the fairness metrics
can be computed on observed latency with no simulation.

```bash
# download the trace with real measured latency (~200 MB)
cd data
curl -L -o BurstGPT_without_fails_3.csv \
  https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_3.csv
cd ..

# compute the quantile-fairness metrics on REAL measured latency
python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort model
python run_burstgpt.py --csv data/BurstGPT_without_fails_3.csv --real-latency --cohort logtype
```

Result on 2,161,220 real requests (observed `Elapsed time`):

| Cohort (B vs A)      | req size B/A | p99 A | p99 B | gap  | ratio | workload-adjusted gap |
|----------------------|:------------:|:-----:|:-----:|:----:|:-----:|:---------------------:|
| GPT-4 vs ChatGPT     | 2.76×        | 48 s  | 68 s  | 20 s | 1.42× | −5.4 s                |
| Conversation vs API  | 1.65×        | 49 s  | 58 s  | 9 s  | 1.18× | −7.3 s                |

The raw tail gaps are real, but the workload-conditioned decomposition shows
they are explained by request-size composition, not per-request scheduling bias
(the adjusted gap is slightly negative). Paired with the synthetic experiment —
the controlled case where scheduling *is* the cause — the metric is validated in
both regimes.

See [docs/DATA.md](docs/DATA.md) for all datasets and licenses, and
[docs/PUBLISHING.md](docs/PUBLISHING.md) for framing.

## Ethics note

Synthetic cohorts ("Group A"/"Group B") are simulated, not real people. On real
traces, `group` is a genuine **operational** field (model, service type, tenant)
— never an inferred proxy for a protected attribute.

## Citation

```bibtex
@article{rai2026taillatency,
  title   = {Tail Latency as an Equity Problem: Quantile Fairness and
             Quality-of-Service Disparities in Serving Systems},
  author  = {Rai, Akhilesh and Bandopadhyay, Sarthak Saptarshi and Chanda, Manash},
  year    = {2026},
  note    = {Preprint. Code: https://github.com/raitheone-svg/tail-latency-equity}
}
```

## License and data credits

Code: MIT, © 2026 Akhilesh Rai, Sarthak Saptarshi Bandopadhyay, Manash Chanda
(see `LICENSE`).

The datasets are **not redistributed here** (they are git-ignored and downloaded
separately) and keep their own licenses. BurstGPT and the Azure Public Dataset
are both CC-BY-4.0; full credits, citations, and license notices are in
[`ATTRIBUTION.md`](ATTRIBUTION.md).
