# data/

Downloaded trace files go here. They are **git-ignored** (too large to commit
and under their own licenses). Fetch them with:

```bash
bash scripts/download_data.sh
```

Then drop the CSV you want into this folder and point `run_trace.py` at it.
See [`../docs/DATA.md`](../docs/DATA.md) for dataset descriptions, licenses,
schemas, and the cohort-definition rules.
