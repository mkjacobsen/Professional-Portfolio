"""
Simple file-based artifact store for models and metrics

Each artifact is stored as a pickled model file alongside a JSON metadata
file. A single ``champion.json`` pointer file tracks which artifact is
currently deployed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

@dataclass
class ArtifactMetadata:
    """
    Metadata record for a stored artifact

    Attributes:
    - artifact_id: Unique identifier string for this artifact
    - artifact_type: One of "model", "features", "metrics", "report
    - version: Monotonically increasing integer version number
    - created_at: ISO-8601 UTC timestamp
    - is_champion: Whether this artifact is the currently deployed champion
    - metadata: Arbitrary key-value pairs
    """

    artifact_id: str
    artifact_type: str
    version: int
    created_at: str
    is_champion: bool
    metadata: dict

class ArtifactStore:
    """
    Persist and retrieve ML artifacts (models, metrics) to/from disk

    Directory layout::
        <store_path>/
            champion.json  <- Points to current champion artifact_id
            model_v1/
                model.pkl
                metadata.json
            model_v2/
                model.pkl
                metadata.json
            ...
    
    Parameters
    - store_path: Root directory for the artifact store.  Created if doesn't exist
    """

    _CHAMPION_FILE = "champion.json"

    def __init__(self, store_path: str | Path) -> None:
        self.path = Path(store_path)
        self.path.mkdir(parents=True, exist_ok=True)


    # Public API
    def save_model(self, model: Any, metrics: dict, version: int) -> str:
        """
        Persist a model and related metrics/data

        Parameters:
        - model: Any picklable sklearn estimator
        - metrics: Dictionary of evaluation metrics
        - version: Integer version number for this artifact

        Returns:
        - str: The ``artifact_id`` that can be passed to :meth:`promote_to_champion`
        """

        artifact_id = f"model_v{version}"
        artifact_dir = self.path / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / "model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        meta = ArtifactMetadata(
            artifact_id = artifact_id,
            artifact_type = "model",
            version = version,
            created_at = datetime.now(timezone.utc).isoformat(),
            is_champion = False,
            metadata = metrics,
        )
        self._write_metadata(meta)
        return artifact_id

    def load_champion(self) -> tuple[Any, dict] | None:
        """
        Load current champion model and its metrics

        Returns:
        - tuple[model, metrics_dict] or None (if no champion has been set)
        """

        champion_id = self._read_champion_id()
        if champion_id is None:
            return None

        artifact_dir = self.path / champion_id
        model_path = artifact_dir / "model.pkl"
        if not model_path.exists():
            return None

        with open(model_path, 'rb') as fh:
            model = pickle.load(fh)

        meta = self._read_metadata(champion_id)
        return model, meta.metadata

    def promote_to_champion(self, artifact_id: str) -> None:
        """
        Designate *artifact_id* as the champion and demote all others.

        Parameters:
        - artifact_id: Artifact id returned by :meth:`save_model`
        """        

        for artifact_dir in self.path.iterdir():
            if not artifact_dir.is_dir():
                continue

            meta_path = artifact_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = self._read_metadata(artifact_dir.name)
            if meta.is_champion:
                meta.is_champion = False
                self._write_metadata(meta)
        
        # Promote specified artifact
        meta = self._read_metadata(artifact_id)
        meta.is_champion = True
        self._write_metadata(meta)

        # Update champion pointer
        champion_path = self.path / self._CHAMPION_FILE
        with open(champion_path, "w") as fh:
            json.dump({"champion_id": artifact_id}, fh)


    def get_champion_mae(self) -> float | None:
        """
        Return MAE of current champion model

        Returs:
        - float or None (if no MAE recorded)
        """

        champion_id = self._read_champion_id()
        if champion_id is None:
            return None

        meta = self._read_metadata(champion_id)
        return meta.metadata.get("mae")

    # Private Helpers

    def _write_metadata(self, meta: ArtifactMetadata) -> None:
        artifact_dir = self.path / meta.artifact_id
        artifact_dir. mkdir(parents=True, exist_ok=True)
        meta_path = artifact_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(asdict(meta), fh, indent=2)

    def _read_metadata(self, artifact_id: str) -> ArtifactMetadata:
        meta_path = self.path / artifact_id / "metadata.json"
        with open(meta_path) as fh:
            data = json.load(fh)
        return ArtifactMetadata(**data)

    def _read_champion_id(self) -> str | None:
        champion_path = self.path / self._CHAMPION_FILE
        if not champion_path.exists():
            return None
        with open(champion_path) as fh:
            data = json.load(fh)
        return data.get("champion_id")