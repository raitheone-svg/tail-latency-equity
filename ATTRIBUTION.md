# Data attribution and credits

This project's **code** is released under the MIT License (see `LICENSE`). The
**datasets** used for validation are not included in this repository (they are
git-ignored and downloaded separately); they remain under their own licenses and
are credited below as their licenses require.

## Datasets used

### BurstGPT — CC-BY-4.0
Used for the real-measured-latency validation (Section 6 of the paper).

- Source: https://github.com/HPMLL/BurstGPT
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
- Citation: Wang, Y., Chen, Y., Li, Z., Kang, X., Fang, Y., Zhou, Y., Zheng, Y.,
  Tang, Z., He, X., Guo, R., Wang, X., Wang, Q., Zhou, A. C., & Chu, X. (2025).
  *BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems.*
  Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data
  Mining (KDD '25). https://doi.org/10.1145/3711896.3737413
- Note: the data is used as-is to compute metrics; no dataset files are
  redistributed in this repository.

### Azure Public Dataset (Azure LLM Inference Trace 2023) — CC-BY-4.0
Used by the simulation-based trace loader.

- Source: https://github.com/Azure/AzurePublicDataset
- License: CC-BY-4.0 (data); MIT (accompanying code)
- Described in: Patel, P., Choukse, E., Zhang, C., Goiri, Í., Shah, A.,
  Maleki, S., & Bianchini, R. (2024). *Splitwise: Efficient Generative LLM
  Inference Using Phase Splitting.* ISCA 2024.

## Also referenced (not required to reproduce the paper)

- **Google cluster-data** (Borg traces): CC-BY, https://github.com/google/cluster-data
- **Alibaba Cluster Trace Program**: research/study use under Alibaba's terms,
  https://github.com/alibaba/clusterdata

## How attribution is satisfied

These datasets are openly licensed (CC-BY-4.0). This project attributes the
creators, links to the original material, states each license, and uses the data
without redistributing the dataset files. Any modifications are limited to
loading and aggregating the data for metric computation, as described in the
paper and in `src/trace_adapters.py`.
