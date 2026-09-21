import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.aiops_pipeline import load_data, run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci


def test_area_of_circle_negative_radius_raises():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)


def test_get_nth_fibonacci_negative_raises():
    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)


def test_get_nth_fibonacci_many_values():
    assert get_nth_fibonacci(2) == 1
    assert get_nth_fibonacci(3) == 2
    assert get_nth_fibonacci(10) == 55


def test_warning_log_is_treated_as_anomaly():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 100,
        "cpu_percent": 40,
        "memory_percent": 48,
        "log_level": "WARNING",
        "message": "Queue warning"
    }

    event = detector.detect(record)

    assert event is not None
    assert event["type"] == "ANOMALY"
    assert "Error log detected" in event["reasons"]


def test_load_data_reads_service_records():
    data = load_data("data/service_data.json")

    assert isinstance(data, list)
    assert len(data) == 10
    assert data[0]["service"] == "payment-service"


def test_run_pipeline_returns_expected_summary():
    result = run_pipeline("data/service_data.json")

    assert result["records_processed"] == 10
    assert len(result["anomalies_detected"]) == 2
    assert result["events_consumed"] == []


def test_aiops_pipeline_script_entrypoint_runs():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [sys.executable, "src/aiops_pipeline.py"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "AIOps Pipeline Result" in result.stdout
    assert "Records processed: 10" in result.stdout
