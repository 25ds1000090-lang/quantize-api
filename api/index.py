import hashlib
import json
import math
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
_lock = threading.RLock()
_freezes: dict[str, dict[str, Any]] = {}

FREEZE_CODES = {
    "INVALID_INPUT", "UNALLOWED_UNSUPPORTED_REASON", "NOT_LOADABLE",
    "CALIBRATION_MISMATCH", "TOKENIZER_MISMATCH",
}


def bad(status: int = 400, code: str = "INVALID_INPUT") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code})


def nonempty_string(value: Any, max_len: int | None = None) -> bool:
    return isinstance(value, str) and len(value) > 0 and (max_len is None or len(value) <= max_len)


def unique_nonempty_strings(value: Any) -> bool:
    return (isinstance(value, list) and all(nonempty_string(x) for x in value)
            and len(value) == len(set(value)))


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def sorted_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=utf8_key)


def files_valid(files: Any) -> bool:
    return (isinstance(files, dict) and len(files) > 0
            and all(nonempty_string(k) and isinstance(v, str) for k, v in files.items()))


def inventory_for(files: dict[str, str]) -> tuple[list[dict[str, Any]], int, str]:
    inventory = []
    for name in sorted(files, key=utf8_key):
        raw = files[name].encode("utf-8")
        inventory.append({"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    total = sum(x["bytes"] for x in inventory)
    return inventory, total, digest_text(compact(inventory))


def freeze_top_valid(data: Any) -> bool:
    if not isinstance(data, dict) or data.get("phase") != "freeze":
        return False
    if not nonempty_string(data.get("freezeId"), 128):
        return False
    if not nonempty_string(data.get("calibrationDigest")) or not nonempty_string(data.get("tokenizerDigest")):
        return False
    if not unique_nonempty_strings(data.get("allowedUnsupportedReasons")):
        return False
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    names = [x.get("name") for x in candidates if isinstance(x, dict)]
    return len(names) == len(candidates) and unique_nonempty_strings(names)


def make_freeze(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(data["allowedUnsupportedReasons"])
    out = []
    for candidate in data["candidates"]:
        reasons: list[str] = []
        files = candidate.get("files")
        if files_valid(files):
            inventory, total, package_digest = inventory_for(files)
        else:
            inventory, total, package_digest = [], None, None
            reasons.append("INVALID_INPUT")

        reason = candidate.get("unsupportedReason")
        # unsupportedReason is either absent/empty (normal candidate), or a non-empty code.
        if reason is not None and not isinstance(reason, str):
            reasons.append("INVALID_INPUT")
        elif isinstance(reason, str) and reason:
            if reason in allowed:
                status = "unsupported" if not reasons else "invalid"
            else:
                reasons.append("UNALLOWED_UNSUPPORTED_REASON")
                status = "invalid"
        else:
            if candidate.get("loadable") is not True:
                reasons.append("NOT_LOADABLE")
            if candidate.get("calibrationDigest") != data["calibrationDigest"]:
                reasons.append("CALIBRATION_MISMATCH")
            if candidate.get("tokenizerDigest") != data["tokenizerDigest"]:
                reasons.append("TOKENIZER_MISMATCH")
            status = "frozen" if not reasons else "invalid"

        if reasons:
            status = "invalid"
        out.append({
            "name": candidate["name"], "status": status, "inventory": inventory,
            "totalBytes": total, "packageDigest": package_digest,
            "reasonCodes": sorted_codes(reasons),
        })
    out.sort(key=lambda x: utf8_key(x["name"]))
    return {"freezeId": data["freezeId"], "candidates": out}


def finite_number(value: Any, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def safe_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9007199254740991


def validate_policy(policy: Any, names: list[str]) -> bool:
    if not isinstance(policy, dict) or not safe_int(policy.get("maxBytes")):
        return False
    if not finite_number(policy.get("aggregateFloor"), 0, 1):
        return False
    if not finite_number(policy.get("maxLatencyMs"), 0):
        return False
    slices = policy.get("requiredSlices")
    if not isinstance(slices, dict) or not all(nonempty_string(k) and finite_number(v, 0, 1) for k, v in slices.items()):
        return False
    order = policy.get("candidateOrder")
    return unique_nonempty_strings(order) and set(order) == set(names)


def valid_manifest(candidate: Any) -> tuple[bool, int | None]:
    if not isinstance(candidate, dict):
        return False, None
    inv = candidate.get("inventory")
    if not isinstance(inv, list) or not inv:
        return False, None
    last = None
    total = 0
    for item in inv:
        if not isinstance(item, dict) or list(item.keys()) != ["name", "bytes", "sha256"]:
            return False, None
        name, size, sha = item.get("name"), item.get("bytes"), item.get("sha256")
        if not nonempty_string(name) or not safe_int(size) or not isinstance(sha, str) or len(sha) != 64:
            return False, None
        try:
            int(sha, 16)
        except ValueError:
            return False, None
        if sha.lower() != sha or (last is not None and utf8_key(last) >= utf8_key(name)):
            return False, None
        last = name
        total += size
        if total > 9007199254740991:
            return False, None
    if candidate.get("totalBytes") != total or candidate.get("packageDigest") != digest_text(compact(inv)):
        return False, total
    return True, total


def rounded_accuracy(correct: int, count: int) -> float:
    return round(correct / count, 12)


def make_select(data: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    submitted = data["candidates"]
    recorded_candidates = recorded["response"]["candidates"]
    names = [x.get("name") for x in submitted if isinstance(x, dict)]
    policy = data["policy"]
    policy_ok = len(names) == len(submitted) and unique_nonempty_strings(names) and validate_policy(policy, names)
    order = policy.get("candidateOrder") if isinstance(policy, dict) and isinstance(policy.get("candidateOrder"), list) else []
    order_index = {name: i for i, name in enumerate(order)}
    rec_by_name = {x["name"]: x for x in recorded_candidates}
    lineage_ok = submitted == recorded_candidates
    latencies = data.get("latencies")
    rows = data["rows"]
    required = policy.get("requiredSlices", {}) if isinstance(policy, dict) else {}
    results = []

    for candidate in sorted(submitted, key=lambda x: (order_index.get(x.get("name"), len(order)), utf8_key(str(x.get("name", ""))))):
        name = candidate.get("name") if isinstance(candidate, dict) else ""
        codes: list[str] = []
        recorded_candidate = rec_by_name.get(name)
        if not lineage_ok or recorded_candidate is None:
            codes.append("INVALID_LINEAGE")
        if recorded_candidate is None or recorded_candidate.get("status") != "frozen":
            codes.append("NOT_FROZEN")
        manifest_ok, recomputed_total = valid_manifest(candidate)
        if not manifest_ok:
            codes.append("INVALID_MANIFEST")
        total = recomputed_total if manifest_ok else None
        if not policy_ok:
            codes.append("INVALID_POLICY")

        latency = latencies.get(name) if isinstance(latencies, dict) else None
        if not finite_number(latency, 0):
            latency = None
            codes.append("INVALID_POLICY")

        predictions_ok = True
        aggregate = None
        slice_values = {slice_name: None for slice_name in required}
        correct = 0
        slice_counts = {slice_name: 0 for slice_name in required}
        slice_correct = {slice_name: 0 for slice_name in required}
        for row in rows:
            if (not isinstance(row, dict) or isinstance(row.get("label"), bool)
                    or row.get("label") not in (0, 1) or not isinstance(row.get("slice"), str)):
                predictions_ok = False
                continue
            predictions = row.get("predictions")
            prediction = predictions.get(name) if isinstance(predictions, dict) else None
            if isinstance(prediction, bool) or prediction not in (0, 1):
                predictions_ok = False
                continue
            correct += int(prediction == row["label"])
            if row["slice"] in required:
                slice_counts[row["slice"]] += 1
                slice_correct[row["slice"]] += int(prediction == row["label"])
        if not rows or not predictions_ok:
            codes.append("INVALID_PREDICTIONS")
        else:
            aggregate = rounded_accuracy(correct, len(rows))
            for slice_name in required:
                if slice_counts[slice_name]:
                    slice_values[slice_name] = rounded_accuracy(slice_correct[slice_name], slice_counts[slice_name])

        if policy_ok and aggregate is not None:
            if aggregate < policy["aggregateFloor"]:
                codes.append("AGGREGATE_FLOOR")
            for slice_name, floor in required.items():
                if slice_counts[slice_name] == 0:
                    codes.append("MISSING_SLICE:" + slice_name)
                elif slice_values[slice_name] < floor:
                    codes.append("SLICE_FLOOR:" + slice_name)
            if total is not None and total > policy["maxBytes"]:
                codes.append("SIZE_LIMIT")
            if latency is not None and latency > policy["maxLatencyMs"]:
                codes.append("LATENCY_LIMIT")

        codes = sorted_codes(codes)
        results.append({"name": name, "aggregate": aggregate, "slices": slice_values,
                        "totalBytes": total, "latencyMs": latency,
                        "admitted": not codes, "reasonCodes": codes})

    admitted = [x for x in results if x["admitted"]]
    winner = min(admitted, key=lambda x: (x["totalBytes"], x["latencyMs"], order_index.get(x["name"], len(order)), utf8_key(x["name"]))) if admitted else None
    winner_manifest = rec_by_name.get(winner["name"]) if winner else None
    return {"freezeId": data["freezeId"], "selected": winner["name"] if winner else None,
            "results": results, "packageManifest": winner_manifest}


@app.post("/quantize")
@app.post("/api/index")
async def quantize(request: Request):
    try:
        data = await request.json()
    except Exception:
        return bad()
    if not isinstance(data, dict) or data.get("phase") not in ("freeze", "select"):
        return bad()

    if data["phase"] == "freeze":
        if not freeze_top_valid(data):
            return bad()
        # Canonical input includes every supplied field and distinguishes JSON types exactly.
        fingerprint = digest_text(compact(data))
        with _lock:
            existing = _freezes.get(data["freezeId"])
            if existing:
                if existing["fingerprint"] != fingerprint:
                    return bad(409, "FREEZE_ID_CONFLICT")
                return existing["response"]
            response = make_freeze(data)
            _freezes[data["freezeId"]] = {"fingerprint": fingerprint, "response": response}
            return response

    if (not nonempty_string(data.get("freezeId"), 128)
            or not isinstance(data.get("candidates"), list)
            or not isinstance(data.get("rows"), list)
            or not isinstance(data.get("policy"), dict)):
        return bad()
    with _lock:
        recorded = _freezes.get(data["freezeId"])
    if recorded is None:
        # Structurally valid selection against an unknown freeze is still evaluated.
        fake = {"response": {"freezeId": data["freezeId"], "candidates": []}}
        return make_select(data, fake)
    return make_select(data, recorded)
