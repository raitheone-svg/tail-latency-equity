# Public datasets for the experiment

The paper's headline results come from a **fully synthetic, seeded simulation**
(`run_synthetic.py`), which needs no downloads. To run the same metrics on real
workloads, use one of the openly-licensed public traces below. The simulation
engine is data-agnostic: a trace only has to supply arrival times and service
demands; the metrics and scheduler are identical.

## A note on "non-licensed" data

Truly licence-free (public-domain / CC0) production serving traces essentially
do not exist. The realistic standard is **openly licensed, free, no-paywall**
data — almost always **CC-BY** (free to use, just cite it). That is accepted by
reviewers and is what the datasets below provide. Always re-check the current
license on the dataset page before use.

## The cohort rule (read before analysing real data)

Real traces carry **no protected attributes**. In this code, `group` is a
neutral **operational** cohort — application, tenant, region, request shape, or
priority — used only to demonstrate the metrics. Never present an operational
split as a demographic one, and never infer a protected attribute. State your
cohort definition explicitly in the paper.

## Datasets

### 1. Azure Public Dataset — CC-BY-4.0 (recommended)
Repo: https://github.com/Azure/AzurePublicDataset  (data CC-BY-4.0, code MIT)

- **AzureFunctionsInvocationTrace2021** — individual function invocations over
  two weeks. Columns: `app, func, end_timestamp, duration`. Use `duration` as
  service demand and `end_timestamp - duration` as arrival. Loader:
  `trace_adapters.load_azure_functions`.
- **AzureLLMInferenceDataset2023** — per-request `TIMESTAMP, ContextTokens,
  GeneratedTokens` for LLM serving. Total tokens approximate service demand.
  Loader: `trace_adapters.load_azure_llm`. **Best choice for novelty** — it puts
  the study in the LLM-serving / TTFT space next to QUARTZ and VTC.

The actual data files are linked from each dataset's `.md` in the repo (cloud
storage). Clone the repo, open the `.md`, download the CSV into `data/`.

### 2. Google cluster-data (Borg) — research use under the repo's terms
Repo: https://github.com/google/cluster-data

- `ClusterData2011_2` — a ~29-day Borg trace, downloadable via `gsutil`.
- `ClusterData2019` — eight Borg cells, **BigQuery-only** (~2.4 TB).

Heavier to work with; good for a datacenter (non-LLM) angle. Follow the access
note in the repo (mailing list + instructions in `ClusterData2011_2.md`).

### 3. Alibaba Cluster Trace Program — study/research use
Repo: https://github.com/alibaba/clusterdata

Microservice and cluster traces; download links appear after a short survey.
Useful for microservice-latency fairness. The underlying data keeps Alibaba's
own terms.

## Quick start on real data

```bash
bash scripts/download_data.sh                       # clones repos + prints links
# ... download a CSV into data/ following the printed instructions ...
python run_trace.py --dataset azure-functions --csv data/AzureFunctionsInvocationTrace2021.csv
python run_trace.py --dataset azure-llm       --csv data/your_llm_trace.csv --w 1.0
```

Confirm each dataset's column names against its own README before running —
schemas can change between releases.
