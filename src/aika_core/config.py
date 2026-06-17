"""Configuration objects for AIKA Core backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.aika_core.data_paths import (
    CLAIMS_FILE,
    ENTITIES_FILE,
    RELATIONS_FILE,
    SEGMENT_DOSSIERS_FILE,
    DEFAULT_CURATED_DIR,
    resolve_data_dir,
)


@dataclass(frozen=True)
class AikaCoreConfig:
    data_dir: Path = DEFAULT_CURATED_DIR
    graph_dir: Path = DEFAULT_CURATED_DIR
    research_dir: Path = DEFAULT_CURATED_DIR
    claims_file: str = CLAIMS_FILE
    entities_file: str = ENTITIES_FILE
    relations_file: str = RELATIONS_FILE
    dossiers_file: str = SEGMENT_DOSSIERS_FILE

    @classmethod
    def from_env(cls) -> "AikaCoreConfig":
        data_dir = resolve_data_dir(os.getenv("AIKA_CORE_DATA_DIR"))
        graph_dir = resolve_data_dir(os.getenv("AIKA_CORE_GRAPH_DIR") or data_dir)
        research_dir = resolve_data_dir(os.getenv("AIKA_CORE_RESEARCH_DIR") or data_dir)
        return cls(data_dir=data_dir, graph_dir=graph_dir, research_dir=research_dir)

    @classmethod
    def from_dir(cls, data_dir: str | Path | None = None) -> "AikaCoreConfig":
        resolved = resolve_data_dir(data_dir)
        return cls(data_dir=resolved, graph_dir=resolved, research_dir=resolved)

    @property
    def claims_path(self) -> Path:
        return self.research_dir / self.claims_file

    @property
    def entities_path(self) -> Path:
        return self.graph_dir / self.entities_file

    @property
    def relations_path(self) -> Path:
        return self.graph_dir / self.relations_file

    @property
    def dossiers_path(self) -> Path:
        return self.research_dir / self.dossiers_file
