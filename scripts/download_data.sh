#!/usr/bin/env bash
# Fetch openly-licensed public traces for the experiment.
# Run from the repo root:   bash scripts/download_data.sh
#
# This script clones the dataset repositories (which hold the schemas and the
# direct download links). Some trace files are hosted on cloud storage and are
# linked from each dataset's own .md file — open those and download the file
# you want into ./data/. See docs/DATA.md for details and licenses.

set -euo pipefail
mkdir -p data/raw
cd data/raw

echo "==> Azure Public Dataset (CC-BY-4.0) ..."
if [ ! -d AzurePublicDataset ]; then
  git clone --depth 1 https://github.com/Azure/AzurePublicDataset.git
fi
echo "    Open AzurePublicDataset/AzureFunctionsInvocationTrace2021.md and"
echo "    AzurePublicDataset/AzureLLMInferenceDataset2023.md for the direct"
echo "    download links, then place the CSV(s) in ../ (the data/ folder)."
echo

echo "==> Google cluster-data (research use; see repo terms) ..."
if [ ! -d cluster-data ]; then
  git clone --depth 1 https://github.com/google/cluster-data.git
fi
echo "    ClusterData2011_2 is downloadable via gsutil; ClusterData2019 is"
echo "    BigQuery-only. Follow cluster-data/ClusterData2011_2.md."
echo

echo "==> Alibaba cluster trace (study/research use) ..."
if [ ! -d alibaba-clusterdata ]; then
  git clone --depth 1 https://github.com/alibaba/clusterdata.git alibaba-clusterdata
fi
echo "    Download links appear after a short survey; see the repo README."
echo
echo "Done. Put the CSV you want to analyse in ./data/ and run, e.g.:"
echo "  python run_trace.py --dataset azure-functions --csv data/AzureFunctionsInvocationTrace2021.csv"
