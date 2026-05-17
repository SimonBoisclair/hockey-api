import os
import sqlite3
import httpx
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

DB_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
DB_PATH = os.path.join(DB_DIR, "app.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            weights TEXT NOT NULL,
            episodes INTEGER DEFAULT 0,
            blue_wins INTEGER DEFAULT 0,
            red_wins INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


class ModelCreate(BaseModel):
    name: str
    weights: str
    episodes: int = 0
    blue_wins: int = 0
    red_wins: int = 0
    draws: int = 0


class ModelUpdate(BaseModel):
    name: str | None = None
    weights: str | None = None
    episodes: int | None = None
    blue_wins: int | None = None
    red_wins: int | None = None
    draws: int | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, episodes, blue_wins, red_wins, draws, created_at, updated_at FROM models ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/models/{model_id}")
async def get_model(model_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return dict(row)


@app.post("/models", status_code=201)
async def create_model(model: ModelCreate):
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO models (name, weights, episodes, blue_wins, red_wins, draws, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (model.name, model.weights, model.episodes, model.blue_wins, model.red_wins, model.draws, now, now),
    )
    conn.commit()
    model_id = cur.lastrowid
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    conn.close()
    return dict(row)


@app.put("/models/{model_id}")
async def update_model(model_id: int, model: ModelUpdate):
    conn = get_db()
    existing = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Model not found")

    now = datetime.now(timezone.utc).isoformat()
    updates = {}
    if model.name is not None:
        updates["name"] = model.name
    if model.weights is not None:
        updates["weights"] = model.weights
    if model.episodes is not None:
        updates["episodes"] = model.episodes
    if model.blue_wins is not None:
        updates["blue_wins"] = model.blue_wins
    if model.red_wins is not None:
        updates["red_wins"] = model.red_wins
    if model.draws is not None:
        updates["draws"] = model.draws
    updates["updated_at"] = now

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [model_id]
    conn.execute(f"UPDATE models SET {set_clause} WHERE id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    conn.close()
    return dict(row)


@app.delete("/models/{model_id}")
async def delete_model(model_id: int):
    conn = get_db()
    existing = conn.execute("SELECT id FROM models WHERE id = ?", (model_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Model not found")
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


# ── RunPod Management ──

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_GQL_URL = "https://api.runpod.io/graphql"
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_URL", "https://hockey-api-zrey.onrender.com")

# In-memory training status (reset on restart, which is fine)
training_status: dict = {}


class TrainingReport(BaseModel):
    episode: int
    total_episodes: int
    blue_wins: int
    red_wins: int
    draws: int
    eps_per_sec: float
    model_name: str
    status: Optional[str] = None
    pod_id: Optional[str] = None


# GPU types to try in order of preference
GPU_FALLBACKS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA RTX 4090",
]


class GPUTrainRequest(BaseModel):
    model_name: str = "gpu-trained"
    episodes: int = 100000
    save_interval: int = 1000
    gpu_type: str = "NVIDIA A100 80GB PCIe"


def runpod_gql(query: str, variables: dict = None) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = httpx.post(
        f"{RUNPOD_GQL_URL}?api_key={RUNPOD_API_KEY}",
        json=payload,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise HTTPException(status_code=400, detail=str(data["errors"]))
    return data.get("data", {})


@app.post("/training/start")
async def start_gpu_training(req: GPUTrainRequest):
    if not RUNPOD_API_KEY:
        raise HTTPException(status_code=500, detail="RUNPOD_API_KEY not configured")

    # Sanitize model name to prevent shell injection
    safe_model_name = req.model_name.replace("'", "").replace('"', '').replace(';', '').replace('&', '').strip()
    if not safe_model_name:
        safe_model_name = "gpu-trained"

    docker_args = (
        f'bash -c "apt-get update && apt-get install -y git && '
        f"rm -rf /workspace/hockey && "
        f"git clone https://github.com/SimonBoisclair/hockey-api.git /workspace/hockey && "
        f"cd /workspace/hockey/training && "
        f"export BACKEND_URL={BACKEND_PUBLIC_URL} && "
        f"export MODEL_NAME='{safe_model_name}' && "
        f"export EPISODES={req.episodes} && "
        f"export SAVE_INTERVAL={req.save_interval} && "
        f"export NUM_ENVS=16384 && "
        f'python train_gpu.py"'
    )

    query_template = """
    mutation {
      podFindAndDeployOnDemand(input: {
        cloudType: ALL,
        gpuCount: 1,
        volumeInGb: 0,
        containerDiskInGb: 20,
        minVcpuCount: 8,
        minMemoryInGb: 32,
        gpuTypeId: "%s",
        name: "hockey-training",
        imageName: "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        dockerArgs: "%s",
        ports: "8080/http"
      }) {
        id
        imageName
        machineId
        costPerHr
      }
    }
    """

    # Try requested GPU type first, then fallbacks
    gpu_types_to_try = [req.gpu_type] + [g for g in GPU_FALLBACKS if g != req.gpu_type]
    pod = None
    used_gpu = req.gpu_type
    last_error = None

    for gpu in gpu_types_to_try:
        try:
            attempt_query = query_template % (gpu, docker_args.replace('"', '\\"'))
            data = runpod_gql(attempt_query)
            pod = data.get("podFindAndDeployOnDemand", {})
            if pod and pod.get("id"):
                used_gpu = gpu
                break
        except Exception as e:
            last_error = str(e)
            continue

    if not pod or not pod.get("id"):
        raise HTTPException(
            status_code=503,
            detail=f"No GPU instances available. Tried: {', '.join(gpu_types_to_try)}. Last error: {last_error}"
        )

    training_status.clear()
    training_status.update({
        "pod_id": pod.get("id"),
        "status": "starting",
        "model_name": req.model_name,
        "total_episodes": req.episodes,
        "episode": 0,
        "blue_wins": 0,
        "red_wins": 0,
        "draws": 0,
        "eps_per_sec": 0,
        "cost_per_hr": pod.get("costPerHr", 0),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "pod_id": pod.get("id"),
        "cost_per_hr": pod.get("costPerHr", 0),
        "status": "starting",
        "model_name": req.model_name,
        "gpu_type": used_gpu,
    }


@app.post("/training/stop")
async def stop_gpu_training():
    pod_id = training_status.get("pod_id")
    if not pod_id:
        raise HTTPException(status_code=404, detail="No active training pod")

    query = 'mutation { podTerminate(input: { podId: "%s" }) }' % pod_id

    try:
        runpod_gql(query)
    except Exception:
        pass

    training_status["status"] = "stopped"
    return {"status": "stopped", "pod_id": pod_id}


@app.post("/training/report")
async def report_training_progress(report: TrainingReport):
    # Only accept reports from the current training pod
    current_pod = training_status.get("pod_id")
    if report.pod_id and current_pod and report.pod_id != current_pod:
        return {"status": "ignored", "reason": "stale pod"}
    training_status.update({
        "status": report.status or "training",
        "episode": report.episode,
        "total_episodes": report.total_episodes,
        "blue_wins": report.blue_wins,
        "red_wins": report.red_wins,
        "draws": report.draws,
        "eps_per_sec": report.eps_per_sec,
        "model_name": report.model_name,
        "last_report": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "ok"}


@app.get("/training/status")
async def get_training_status():
    if not training_status:
        return {"status": "idle"}
    return training_status
