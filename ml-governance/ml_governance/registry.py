"""
registry.py - sqlite-backed model registry.
"""

import json
import pickle
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from .model_card import ModelCard

@dataclass
class RegistryEntry:
    model_id: str
    model_name: str
    version: str
    registered_at: str
    status: str # "staging" | "production" | "archived"
    model_path: str # path to pickled model file
    card_path: str # path to model card JSON
    tags: dict # arbitrary key-value metadata

_VALID_STATUSES = ("staging", "production", "archived")

_CREATE_TABLE_SQL ="""
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    registered at TEXT NOT NULL,
    status TEXT NOT NULL,
    model_path TEXT NOT NULL,
    card_path TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '{}'
)"""

class ModelRegistry:
    """Persist trained models and their model cards in a local SQlite registry."""

    def __init__ (self, registry_path: str | Path):
        self.path = Path(registry_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = self.path / "registry.db"
        self._models_dir = self.path/ "models"
        self._cards_dir = self.path / "carde"
        self._models_dir.mkdir(exist_ok=True)
        self._cards_dir.mkdir(exist_ok=True)
        self._init_db()

    # Internal belpers
    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_entry(self, row: sqlite3.Row) -> RegistryEntry:
        return RegistryEntry(
            model_id = row["model_id"],
            model_name = row["model_name"],
            version = row["version"],
            registered_at = row["registered_at"],
            status = row["status"],
            model_path = row["model_path"],
            card_path = row["card_path"],
            tags = json.loads(row["tags_json"]),
        )

    # Public API

    def register(
            self,
            model: object,
            model_name: str,
            version: str,
            card: "ModelCard",
            tags: dict | None = None,
    ) -> str:
        """
        Pickle *model*, serialize *Card* to JSON, record both in the DB.

        Returns the newly assigned ``model_id``
        """

        model_id = str(uuid.uuid4())
        registered_at = datetime.now(timezone.utc).isoformat()
        tags = tags or {}

        # Persist artifacts
        model_path = self._models_dir / f"{model_id}.pkl"
        card_path = self._cards_dir / f"{model_id}.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Inject model id to complete the model card before saving
        card.model_id = model_id
        card_path.write_text(card.to_json(), encoding="utf-8")

        # Insert into DB
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO models
                    (model_id, model_name, version, registered_at, status,
                     model_path, card_path, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_name,
                    version,
                    registered_at,
                    "staging",
                    str(model_path),
                    str(card_path),
                    json.dumps(tags),
                ),
            )
            conn.commit()

        return model_id

    def get(self, model_id: str) -> tuple[object, ModelCard]:
        """
        Load and return ``(model,card)`` for the given "model_id"

        Raises ``KeyError`` if the model is not found.
        """

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()

        if row is None:
            raise KeyError(f"Model '{model_id}' not found in registry.")

        entry = self._row_to_entry(row)

        with open(entry.model_path, "rb") as f:
            model = pickle.load(f)
            card = ModelCard.from_json(Path(entry.card_path).read_text(encoding="utf-8"))
            return model, card

    def list_models(self, status: str | None = None) -> list[RegistryEntry]:
        """
        Return all registry entries, optionally filtered by *status*
        """
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM models WHERE status = ? ORDER BY registered_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM models ORDER BY registered_at DESC"
                ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def update_status(self, model_id: str, status: str) -> None:
        """
        Transition a model to a new lifecycle status.

        Valid values: ``"staging"``, ``"production"``, ``"archived"``
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of {_VALID_STATUSES}"
            )

        with self._connect() as conn:
            result = conn.execute(
                "UPDATE models SET status = ? WHERE model_id = ?",
                (status, model_id),
            )
            conn.commit()

        if result.rowcount == 0:
            raise KeyError(f"Model '{model_id}' not found in registry.")

    def compare(self, model_id_a: str, model_id_b: str) -> dict:
        """
        Load both model cards and return side-by-side comparison of key metrics.
        
        Returns

        {
            "model_a": {"name":..., "version":..., "gini":..., ...},
            "model_b": {...},
            "delta": {"gini": ..., "roc_auc": ..., ...},
        }
        """

        _, card_a = self.get(model_id_a)
        _, card_b = self.get(model_id_b)

        def _metrics_snapshot(card: ModelCard) -> dict:
            p = card.performance
            return {
                "name": card.model_name,
                "version": card.version,
                "accuracy": p.accuracy,
                "precision": p.precision,
                "recall": p.recall,
                "f1": p.f1,
                "roc_auc": p.roc_auc,
                "gini": p.gini,
                "ks_statistic": p.ks_statistic,
                "risk_score": card.risk_score,
            }

        snap_a = _metrics_snapshot(card_a)
        snap_b = _metrics_snapshot(card_b)

        numeric_keys = [
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "gini", "ks_statistic", "risk_score"
        ]

        delta = {
            k: (snap_b[k] - snap_a[k])
            if (snap_a[k] is not None and snap_b[k] is not None)
            else None
            for k in numeric_keys
        }

        return {"model_a": snap_a, "model_b": snap_b, "delta": delta}

