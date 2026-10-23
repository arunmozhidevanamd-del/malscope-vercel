# MalScope — Malware Behavior Profiling

A Vercel-ready web app with a FastAPI backend and interactive browser frontend. It groups numeric behavioral telemetry using `StandardScaler` + `KMeans`, visualizes the clusters with PCA, reports cluster sizes and silhouette score, summarizes cluster profiles, supports CSV upload, and exports assignments to CSV.

## Deploy on Vercel from GitHub

1. Upload the contents of this folder to the **root** of a GitHub repository.
2. Visit https://vercel.com/new and import that repository.
3. Keep the detected Python/FastAPI settings and deploy. The API entry point is root-level `app.py`; static frontend files are in `public/`.
4. Test the live homepage and `/api/health`.

No environment variables are required.

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## CSV schema

CSV must contain all these numeric columns:

`api_calls, files_created, files_modified, registry_changes, network_connections, dns_queries, processes_spawned, cpu_usage, memory_usage, entropy`

Optional `sample_id` column is shown in the assignments table. Rows with missing or invalid feature values and duplicate feature profiles are removed.

## Important

The built-in dataset is **synthetic demonstration data**, not real malware telemetry. This is a behavior-clustering educational prototype, not a malware verdict or a substitute for professional security analysis. The app does not execute uploaded files; upload CSV telemetry only.
