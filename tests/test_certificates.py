"""Batch certificate endpoint tests.

Creation reuses the release verification: only a RELEASED verdict is
certified, freezing layout/scan/readings/thresholds and the blinded-view
digest into one SQLite transaction. Re-verification restores the frozen
parameters, increments the revision atomically on success and revokes the
certificate on any misplacement, QC failure or view-digest change (first
failure reason returned). A revoked certificate can never be restored and
stale revisions can never overwrite. Idempotency: the same request_key with
identical content replays the recorded response; different content is 409.
"""
import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import certificates
from app.main import app

client = TestClient(app)

# 4 rows x 4 cols, 16 wells, two groups interleaved checkerboard-style;
# integer noise gives small, well-converged row/column effects.
LAYOUT_4X4 = {
    "gA1": "A1", "gA2": "A3", "gB1": "A2", "gB2": "A4",
    "gB3": "B1", "gB4": "B3", "gA3": "B2", "gA4": "B4",
    "gA5": "C1", "gA6": "C3", "gB5": "C2", "gB6": "C4",
    "gB7": "D1", "gB8": "D3", "gA7": "D2", "gA8": "D4",
}
SAMPLES_4X4 = [
    {"id": s, "group": "GA" if s.startswith("gA") else "GB"}
    for s in LAYOUT_4X4
]
PLATE_4X4 = {"type": "custom", "rows": 4, "columns": 4}


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    """Isolate every test in its own SQLite database."""
    monkeypatch.setenv("CERT_DB_PATH", str(tmp_path / "certificates.db"))
    certificates.reset_stores()
    yield tmp_path / "certificates.db"
    certificates.reset_stores()


def _hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _readings(row_bump=0.0):
    out = {}
    for sid, well in LAYOUT_4X4.items():
        base = 100.0 if sid.startswith("gA") else 0.0
        noise = (_hash(sid) % 7) - 3  # -3..3
        out[well] = base + (row_bump if well[0] == "A" else 0.0) + noise
    return out


def _release_payload(**over):
    payload = {
        "plate": PLATE_4X4,
        "target": dict(LAYOUT_4X4),
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "blocked_wells": [],
        "samples": [dict(s) for s in SAMPLES_4X4],
        "readings": _readings(),
        "seed": 7,
        "max_spatial_effect": 10.0,
        "min_p_value": 0.001,
        "permutations": 100,
        "tolerance": 1e-6,
        "max_iterations": 100,
    }
    payload.update(over)
    return payload


def _view():
    """Blinded operational view over the layout wells (row-major order)."""
    wells = sorted(LAYOUT_4X4.values(), key=lambda w: (w[0], int(w[1:])))
    return [{"well": w, "blind_id": f"BL-{i:02d}"} for i, w in enumerate(wells)]


def _create_payload(**over):
    payload = {
        "batch_id": "BATCH-1",
        "request_key": "create-1",
        "release": _release_payload(),
        "view": _view(),
    }
    payload.update(over)
    return payload


def _reverify_payload(**over):
    payload = {
        "batch_id": "BATCH-1",
        "request_key": "reverify-1",
        "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": _readings(),
        "view": _view(),
    }
    payload.update(over)
    return payload


def _create_cert(**over):
    r = client.post("/api/v1/certificates", json=_create_payload(**over))
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------- create


def test_create_success_returns_certificate_fields():
    r = client.post("/api/v1/certificates", json=_create_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["certificate_id"].startswith("cert-")
    assert body["batch_id"] == "BATCH-1"
    assert body["status"] == "ACTIVE"
    assert body["revision"] == 1
    assert len(body["digest"]) == 64 and int(body["digest"], 16) >= 0
    assert len(body["view_digest"]) == 64


def test_create_without_view_has_null_view_digest():
    payload = _create_payload()
    del payload["view"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "ACTIVE"
    assert body["view_digest"] is None


def test_create_not_released_creates_no_certificate():
    payload = _create_payload()
    payload["release"]["max_spatial_effect"] = 0.01  # effect gate fails
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["certificate_id"] is None
    assert body["status"] == "NOT_CERTIFIED"
    assert body["verdict"] == "NOT_RELEASED"
    assert body["reasons"] == ["SPATIAL_EFFECT_EXCEEDED"]
    # Nothing was persisted: re-verification finds no certificate.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"


def test_create_scan_mismatch_is_422_and_creates_nothing():
    payload = _create_payload()
    del payload["release"]["scan"]["A1"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert "SAMPLES_MISSING" in body["infeasibility_categories"]
    assert body["misplaced_wells"] == [
        {"well": "A1", "expected": "gA1", "actual": None}]
    assert "certificate_id" not in body
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404


def test_create_replay_same_key_same_content_returns_recorded_response():
    payload = _create_payload()
    b1 = client.post("/api/v1/certificates", json=payload)
    b2 = client.post("/api/v1/certificates", json=payload)
    assert b1.status_code == 201 and b2.status_code == 201
    assert b1.json() == b2.json()


def test_create_same_key_different_content_is_409():
    _create_cert()
    payload = _create_payload()
    payload["release"]["seed"] = 8  # same request_key, different content
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "REQUEST_KEY_CONFLICT"
    assert body["request_key"] == "create-1"


def test_create_second_certificate_for_same_batch_is_409():
    first = _create_cert()
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_ALREADY_EXISTS"
    assert body["certificate_id"] == first["certificate_id"]
    assert body["status"] == "ACTIVE"


def test_create_invalid_view_is_422_and_creates_nothing():
    payload = _create_payload()
    payload["view"][1]["blind_id"] = payload["view"][0]["blind_id"]  # duplicate
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert "CERTIFICATE_VIEW_INVALID" in r.json()["infeasibility_categories"]

    payload = _create_payload(request_key="create-2")
    payload["view"][0]["well"] = "Z9"  # outside the plate
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]

    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404


def test_create_not_certified_response_is_recorded_under_the_key():
    payload = _create_payload()
    payload["release"]["max_spatial_effect"] = 0.01
    b1 = client.post("/api/v1/certificates", json=payload)
    b2 = client.post("/api/v1/certificates", json=payload)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()
    # Same key, different content -> 409; fresh key -> the batch can certify.
    payload["release"]["seed"] = 8
    assert client.post("/api/v1/certificates", json=payload).status_code == 409
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 201


# ------------------------------------------------------------------- reverify


def test_reverify_success_increments_revision_atomically():
    created = _create_cert()
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ACTIVE"
    assert body["revision"] == 2
    assert body["certificate_id"] == created["certificate_id"]
    assert body["digest"] == created["digest"]
    assert body["view_digest"] == created["view_digest"]

    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=2))
    assert r.status_code == 200
    assert r.json()["revision"] == 3


def test_reverify_replay_does_not_double_increment():
    _create_cert()
    payload = _reverify_payload()
    b1 = client.post("/api/v1/certificates/reverify", json=payload)
    b2 = client.post("/api/v1/certificates/reverify", json=payload)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()
    assert b2.json()["revision"] == 2
    # The replay did not burn a revision: 2 is still the current one.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=2))
    assert r.status_code == 200
    assert r.json()["revision"] == 3


def test_reverify_same_key_different_content_is_409():
    _create_cert()
    client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    payload = _reverify_payload(expected_revision=2)
    payload["readings"] = _readings(row_bump=0.5)  # same key, new content
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


def test_reverify_stale_revision_is_409_and_never_overwrites():
    _create_cert()
    client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    # Current revision is 2; resubmitting against the old revision 1 fails.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2", expected_revision=1))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert body["current_revision"] == 2
    assert body["expected_revision"] == 1
    # The certificate is untouched: the correct revision still verifies.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-3", expected_revision=2))
    assert r.status_code == 200
    assert r.json()["revision"] == 3


def test_reverify_unknown_batch_is_404():
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"


def test_reverify_closed_loop_misplacement_revokes_with_first_reason():
    created = _create_cert()
    payload = _reverify_payload()
    payload["scan"]["A1"], payload["scan"]["A2"] = "gB1", "gA1"  # swap
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revision"] == 1  # never incremented
    assert body["digest"] == created["digest"]
    revocation = body["revocation"]
    assert revocation["reason"] == "CLOSED_LOOP_MISPLACEMENT"
    assert revocation["cycles"] == [["A1", "A2"]]
    assert {"well": "A1", "expected": "gA1", "actual": "gB1"} in \
        revocation["misplaced_wells"]


def test_reverify_missing_sample_revokes():
    _create_cert()
    payload = _reverify_payload()
    del payload["scan"]["A1"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "SAMPLES_MISSING"


def test_reverify_qc_gate_failure_revokes():
    _create_cert()
    # Fresh readings carry a strong row-A effect that breaches the frozen
    # max_spatial_effect=10.0 threshold.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        readings=_readings(row_bump=100.0)))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "SPATIAL_EFFECT_EXCEEDED"


def test_reverify_structural_reading_problem_revokes():
    _create_cert()
    payload = _reverify_payload()
    payload["readings"]["a1"] = 1.0  # duplicates "A1" after normalisation
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "REVOKED"
    assert r.json()["revocation"]["reason"] == "SPATIAL_READING_DUPLICATED"


def test_reverify_view_digest_change_revokes():
    _create_cert()
    payload = _reverify_payload()
    payload["view"][0]["blind_id"] = "BL-tampered"
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revocation"]["reason"] == "VIEW_DIGEST_MISMATCH"


def test_reverify_with_matching_view_passes():
    _create_cert()
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload())
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"


def test_reverify_without_view_passes():
    _create_cert()  # certificate carries a view digest
    payload = _reverify_payload()
    del payload["view"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"


def test_reverify_invalid_view_is_422_and_leaves_certificate_active():
    _create_cert()
    payload = _reverify_payload()
    payload["view"][0]["well"] = "Z9"
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]
    # Not a verification failure: the certificate is still ACTIVE at rev 1.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2"))
    assert r.status_code == 200
    assert r.json()["revision"] == 2


def test_revoked_certificate_can_never_be_restored():
    _create_cert()
    payload = _reverify_payload()
    del payload["scan"]["A1"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.json()["status"] == "REVOKED"

    # Replaying the revoking request returns the recorded REVOKED response.
    replay = client.post("/api/v1/certificates/reverify", json=payload)
    assert replay.status_code == 200
    assert replay.json() == r.json()

    # Any new re-verification is refused; revocation is final.
    r = client.post("/api/v1/certificates/reverify", json=_reverify_payload(
        request_key="reverify-2"))
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["revoked_reason"] == "SAMPLES_MISSING"

    # And no new certificate can be created for the batch either.
    r = client.post("/api/v1/certificates",
                    json=_create_payload(request_key="create-2"))
    assert r.status_code == 409
    assert r.json()["error"] == "CERTIFICATE_ALREADY_EXISTS"
    assert r.json()["status"] == "REVOKED"


# ------------------------------------------------------------- confidentiality


def test_certificate_stores_no_view_entries_or_plaintext_blind_mapping(fresh_store):
    _create_cert()
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot, view_digest, digest FROM certificates").fetchone()
    conn.close()
    snapshot, view_digest, digest = row
    stored = json.loads(snapshot)
    # The snapshot freezes layout/scan/readings/thresholds plus the view
    # DIGEST only — never the view entries (well -> blind_id) themselves.
    assert stored["view_digest"] == view_digest
    assert "view" not in stored
    assert set(stored) == {
        "batch_id", "plate", "target", "scan", "readings", "samples",
        "blocked_wells", "seed", "thresholds", "view_digest",
    }
    assert len(view_digest) == 64 and len(digest) == 64
    # No blind_id appears anywhere in the database files (main db or WAL).
    for suffix in ("", "-wal", "-shm"):
        path = fresh_store.with_name(fresh_store.name + suffix)
        if path.exists():
            assert b"BL-" not in path.read_bytes()


# ------------------------------------------------------------ request shape


def test_missing_request_key_is_request_validation_error():
    payload = _create_payload()
    del payload["request_key"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_reverify_missing_expected_revision_is_request_validation_error():
    payload = _reverify_payload()
    del payload["expected_revision"]
    r = client.post("/api/v1/certificates/reverify", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_unknown_field_is_422():
    payload = _create_payload()
    payload["mapping_package"] = "secret"  # not part of the certificate contract
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_existing_endpoints_remain_available():
    r = client.post("/api/v1/arrangements/batch-release", json=_release_payload())
    assert r.status_code == 200
    assert r.json()["verdict"] == "RELEASED"
    assert client.get("/health").json() == {"status": "ok"}
