"""バックテスト再現用manifest。

出力ディレクトリに以下を保存する:
  - manifest.json: git commit, config SHA256, データ fingerprint, Python version
  - config.resolved.json: 解決済み設定（シークレットはREDACTED）
  - environment.txt: pip freeze
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SECRET_KEYWORDS = ("api_key", "token", "password", "webhook", "secret")


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None

    p = Path(path)
    if not p.exists() or not p.is_file():
        return None

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return sorted(line.strip() for line in out.splitlines() if line.strip())
    except Exception:
        return []


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k).lower()
            if any(s in key for s in _SECRET_KEYWORDS):
                out[k] = "REDACTED"
            else:
                out[k] = _redact(v)
        return out

    if isinstance(obj, list):
        return [_redact(v) for v in obj]

    return obj


def dataframe_fingerprint(df: pd.DataFrame | None) -> str | None:
    """DataFrame内容の簡易fingerprint。"""
    if df is None or df.empty:
        return None

    x = df.copy()
    x = x.sort_index(axis=1)
    csv = x.to_csv(index=False)
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def write_backtest_manifest(
    *,
    out_dir: str | Path,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    prices: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    shortability: pd.DataFrame | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """バックテスト再現用manifestを書き出す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": _sha256_file(config_path),
        "prices_fingerprint": dataframe_fingerprint(prices),
        "universe_fingerprint": dataframe_fingerprint(universe),
        "shortability_fingerprint": dataframe_fingerprint(shortability),
        "extra": extra or {},
    }

    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    with (out / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(
            _redact(config),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

    with (out / "environment.txt").open("w", encoding="utf-8") as f:
        for line in _pip_freeze():
            f.write(line + "\n")
