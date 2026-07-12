"""インパクト較正アーティファクト管理（PR-3）。

ImpactModelArtifact で較正済みインパクト係数を永続化・読み出しする。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    """ファイルの SHA256 を計算する。"""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ImpactModelArtifact:
    """較正済みインパクトモデルのアーティファクト。"""

    version: str
    k_bp: float
    adv_window: int
    min_adv_periods: int
    fills_md5: str
    model_type: str = "square_root"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "k_bp": self.k_bp,
            "adv_window": self.adv_window,
            "min_adv_periods": self.min_adv_periods,
            "fills_md5": self.fills_md5,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "description": self.description,
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info("ImpactModelArtifact saved to %s (k_bp=%.2f)", p, self.k_bp)

    @classmethod
    def load(cls, path: str | Path) -> ImpactModelArtifact:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ImpactModelArtifact not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            version=str(raw["version"]),
            k_bp=float(raw["k_bp"]),
            adv_window=int(raw["adv_window"]),
            min_adv_periods=int(raw["min_adv_periods"]),
            fills_md5=str(raw["fills_md5"]),
            model_type=str(raw.get("model_type", "square_root")),
            created_at=str(raw.get("created_at", "")),
            description=str(raw.get("description", "")),
            parameters=raw.get("parameters", {}),
            diagnostics=raw.get("diagnostics", {}),
        )


def load_impact_artifact(
    path: str | Path,
    *,
    expected_fills_md5: str | None = None,
    warn_on_mismatch: bool = True,
) -> ImpactModelArtifact:
    """ImpactModelArtifact を読み込み、必要に応じてフィンガープリントを検証する。"""
    artifact = ImpactModelArtifact.load(path)
    if expected_fills_md5 is not None and artifact.fills_md5 != expected_fills_md5:
        msg = (
            f"fills MD5 mismatch: artifact={artifact.fills_md5}, "
            f"expected={expected_fills_md5}"
        )
        if warn_on_mismatch:
            log.warning(msg)
        else:
            raise ValueError(msg)
    return artifact
