"""Ingest stage result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class IngestStatus(str, Enum):
    """Outcome of a single ingest operation."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestRecord:
    """Record for one ingested source file."""

    source_path: str
    format: str
    output_dir: str | None
    status: IngestStatus
    message: str
    layer_hint: str | None = None
    member_count: int = 0
    members_sample: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON ingest logs."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class IngestReport:
    """Aggregated ingest run output."""

    format: str
    records: list[IngestRecord] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        """Number of successfully ingested sources."""
        return sum(1 for r in self.records if r.status == IngestStatus.OK)

    @property
    def failed_count(self) -> int:
        """Number of failed ingest operations."""
        return sum(1 for r in self.records if r.status == IngestStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report for JSON export."""
        return {
            "format": self.format,
            "ok_count": self.ok_count,
            "failed_count": self.failed_count,
            "records": [r.to_dict() for r in self.records],
        }
