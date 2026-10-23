from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
import io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "api_calls", "files_created", "files_modified", "registry_changes",
    "network_connections", "dns_queries", "processes_spawned",
    "cpu_usage", "memory_usage", "entropy",
]
app = FastAPI(title="MalScope API", version="1.0.0",
              description="Malware behavior profiling demo using K-Means and PCA.")



class AnalyzeRequest(BaseModel):
    k: int = Field(default=5, ge=2, le=8)
    demo_samples: int = Field(default=600, ge=100, le=1500)


def demo_data(n=600, seed=42):
    """Synthetic demonstration telemetry; it is not real malware data."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [20, 1, 2, 1, 1, 1, 2, 12, 18, 4.2],
        [75, 8, 15, 12, 22, 18, 10, 78, 72, 7.1],
        [48, 3, 7, 5, 35, 32, 7, 45, 55, 6.2],
        [35, 14, 18, 22, 4, 2, 18, 65, 80, 5.8],
        [90, 2, 4, 3, 12, 8, 4, 30, 38, 7.7],
    ], dtype=float)
    labels = rng.integers(0, len(centers), size=n)
    scales = np.array([8, 2.5, 3.5, 3.5, 5, 5, 3, 9, 9, .35])
    values = centers[labels] + rng.normal(0, scales, size=(n, len(FEATURES)))
    for idx in range(7):
        values[:, idx] = np.clip(np.rint(values[:, idx]), 0, None)
    values[:, 7:9] = np.clip(values[:, 7:9], 0, 100)
    values[:, 9] = np.clip(values[:, 9], 0, 8)
    df = pd.DataFrame(values, columns=FEATURES)
    df.insert(0, "sample_id", [f"SYN-{i+1:04d}" for i in range(n)])
    df.insert(1, "source", "Synthetic demo")
    return df


def analyze_frame(raw: pd.DataFrame, k: int):
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    original_rows = len(raw)
    data = raw.copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES).drop_duplicates(subset=FEATURES).reset_index(drop=True)
    if len(data) < 3:
        raise ValueError("At least 3 valid, non-duplicate rows are required.")
    if all(data[c].nunique() == 1 for c in FEATURES):
        raise ValueError("All behavioral feature values are identical; clustering is not possible.")
    k = max(2, min(int(k), 8, len(data) - 1))
    X = data[FEATURES].astype(float)
    scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
    unique_labels = np.unique(labels)
    score = silhouette_score(scaled, labels) if 1 < len(unique_labels) < len(data) else float("nan")
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled)
    data["cluster"] = labels.astype(int)
    profiles = data.groupby("cluster")[FEATURES].mean().round(2)
    scaled_profile = pd.DataFrame(scaled, columns=FEATURES).assign(cluster=labels).groupby("cluster").mean()
    dominant = {}
    for cid, row in scaled_profile.iterrows():
        dominant[str(int(cid))] = row.sort_values(ascending=False).head(3).index.tolist()
    counts = data["cluster"].value_counts().sort_index()
    assignments = []
    for i, row in data.iterrows():
        item = {"sample_id": str(row.get("sample_id", f"Sample {i+1}")),
                "cluster": int(row["cluster"])}
        for col in FEATURES:
            item[col] = float(row[col])
        assignments.append(item)
    return {
        "samples": len(data),
        "input_rows": original_rows,
        "rows_removed": int(original_rows - len(data)),
        "clusters": int(len(unique_labels)),
        "features_used": len(FEATURES),
        "silhouette_score": None if not np.isfinite(score) else round(float(score), 4),
        "pca_variance_percent": round(float(pca.explained_variance_ratio_.sum() * 100), 1),
        "features": FEATURES,
        "points": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                    "cluster": int(labels[i]), "sample_id": str(data.iloc[i].get("sample_id", f"Sample {i+1}"))}
                   for i, p in enumerate(points)],
        "counts": [{"cluster": int(cid), "count": int(count)} for cid, count in counts.items()],
        "profiles": [{"cluster": int(cid), **{f: float(v) for f, v in row.items()}}
                     for cid, row in profiles.iterrows()],
        "dominant_behaviors": dominant,
        "assignments": assignments,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "MalScope"}


@app.post("/api/analyze-demo")
def analyze_demo(request: AnalyzeRequest):
    try:
        return analyze_frame(demo_data(request.demo_samples), request.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...), k: int = 5):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    if k < 2 or k > 8:
        raise HTTPException(status_code=400, detail="Cluster count k must be between 2 and 8.")
    try:
        content = await file.read()
        if len(content) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="CSV is too large. Maximum upload size is 8 MB.")
        raw = pd.read_csv(io.BytesIO(content))
        return analyze_frame(raw, k)
    except HTTPException:
        raise
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Could not read CSV.")


@app.post("/api/export-csv")
def export_csv(payload: dict):
    rows = payload.get("assignments", [])
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="No clustered results to export.")
    frame = pd.DataFrame(rows)
    csv = frame.to_csv(index=False)
    return Response(
        content=csv,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="malscope_clustered_results.csv"'}
    )

@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse("index.html")
