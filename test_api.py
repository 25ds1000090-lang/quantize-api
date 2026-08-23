from fastapi.testclient import TestClient
from api.index import app

c = TestClient(app)


def test_freeze_replay_conflict_and_select():
    freeze = {
        "phase": "freeze", "freezeId": "demo", "calibrationDigest": "cal",
        "tokenizerDigest": "tok", "allowedUnsupportedReasons": [],
        "candidates": [
            {"name": "int8", "files": {"model.safetensors": "abc"}, "loadable": True,
             "calibrationDigest": "cal", "tokenizerDigest": "tok", "unsupportedReason": ""},
            {"name": "int4", "files": {"model.safetensors": "a"}, "loadable": True,
             "calibrationDigest": "cal", "tokenizerDigest": "tok", "unsupportedReason": ""},
        ],
    }
    first = c.post("/quantize", json=freeze)
    assert first.status_code == 200
    assert first.json()["candidates"][0]["name"] == "int4"
    assert c.post("/quantize", json=freeze).json() == first.json()
    changed = {**freeze, "tokenizerDigest": "changed"}
    assert c.post("/quantize", json=changed).status_code == 409

    frozen = first.json()["candidates"]
    select = {
        "phase": "select", "freezeId": "demo", "candidates": frozen,
        "policy": {"maxBytes": 10, "aggregateFloor": .5, "requiredSlices": {"critical": .5},
                   "maxLatencyMs": 100, "candidateOrder": ["int8", "int4"]},
        "latencies": {"int8": 60, "int4": 40},
        "rows": [{"label": 1, "slice": "critical", "predictions": {"int8": 1, "int4": 0}}],
    }
    result = c.post("/quantize", json=select).json()
    assert result["selected"] == "int8"
    assert result["packageManifest"]["name"] == "int8"

