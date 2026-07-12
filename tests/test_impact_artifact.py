"""Tests for impact_artifact module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from jp_signal.impact_artifact import (
    ImpactModelArtifact,
    load_impact_artifact,
    sha256_file,
)


def test_sha256_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello")
        path = f.name
    try:
        digest = sha256_file(path)
        assert isinstance(digest, str)
        assert len(digest) == 64
    finally:
        Path(path).unlink(missing_ok=True)


def test_impact_model_artifact_save_and_load():
    artifact = ImpactModelArtifact(
        version="1.0",
        k_bp=30.0,
        adv_window=20,
        min_adv_periods=20,
        fills_md5="abc123",
        model_type="square_root",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        artifact.save(path)
        loaded = ImpactModelArtifact.load(path)
        assert loaded.k_bp == 30.0
        assert loaded.version == "1.0"
        assert loaded.fills_md5 == "abc123"
        assert loaded.model_type == "square_root"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_impact_artifact_not_found():
    with pytest.raises(FileNotFoundError):
        ImpactModelArtifact.load("/nonexistent/artifact.json")


def test_load_impact_artifact_md5_mismatch_warn():
    artifact = ImpactModelArtifact(
        version="1.0",
        k_bp=30.0,
        adv_window=20,
        min_adv_periods=20,
        fills_md5="abc123",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        artifact.save(path)
        loaded = load_impact_artifact(path, expected_fills_md5="xyz789", warn_on_mismatch=True)
        assert loaded.fills_md5 == "abc123"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_impact_artifact_md5_mismatch_strict():
    artifact = ImpactModelArtifact(
        version="1.0",
        k_bp=30.0,
        adv_window=20,
        min_adv_periods=20,
        fills_md5="abc123",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        artifact.save(path)
        with pytest.raises(ValueError, match="fills MD5 mismatch"):
            load_impact_artifact(path, expected_fills_md5="xyz789", warn_on_mismatch=False)
    finally:
        Path(path).unlink(missing_ok=True)


def test_impact_model_artifact_with_params():
    artifact = ImpactModelArtifact(
        version="2.0",
        k_bp=25.0,
        adv_window=40,
        min_adv_periods=10,
        fills_md5="def456",
        parameters={"max_iter": 100, "tol": 1e-4},
        diagnostics={"r2": 0.95, "residual_std": 0.02},
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        artifact.save(path)
        loaded = ImpactModelArtifact.load(path)
        assert loaded.parameters == {"max_iter": 100, "tol": 1e-4}
        assert loaded.diagnostics["r2"] == 0.95
    finally:
        Path(path).unlink(missing_ok=True)
