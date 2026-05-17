import os
import sqlite3
from datetime import datetime, timezone

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
