"""Release certificate endpoint tests.

Covers:
* creation only persists on RELEASED (201) — NOT_RELEASED verdicts create no
  certificate but remain replayable;
* the single SQLite transaction stores layout/scan/readings/thresholds and
  only a non-invertible view summary — never key shares or blind mappings;
* re-verification restores frozen parameters, increments revision atomically
  with a fresh digest, revokes irreversibly on scan mismatch / QC gate
  failure / view summary change, and reports the first failing reason;
* REVOKED is terminal, old revisions cannot be overwritten;
* request_key replays same content verbatim and 409s on changed content
  (including cross-operation reuse), unknown batch is 404.
"""
from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Same 4x4 fixture family as tests/test_release.py: interleaved groups with
# small integer noise, well-converged, p-value friendly.
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


def _hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _readings(row_bump_wells=(), bump=0.0):
    out = {}
    for sid, well in LAYOUT_4X4.items():
        base = 100.0 if sid.startswith("gA") else 0.0
        noise = (_hash(sid) % 7) - 3
        out[well] = base + (bump if well in row_bump_wells else 0.0) + noise
    return out


def _create_payload(**over):
    payload = {
        "plate": PLATE_4X4,
        "batch_id": "LOT-1",
        "request_key": "create-1",
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


def _reverify_payload(**over):
    payload = {
        "batch_id": "LOT-1",
        "request_key": "reverify-1",
        "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": _readings(),
    }
    payload.update(over)
    return payload


def _certify(batch_id="LOT-1", request_key="create-1", **over):
    body = _create_payload(batch_id=batch_id, request_key=request_key, **over)
    r = client.post("/api/v1/certificates", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------- create 201


def test_evidence_digest_is_deterministic_and_binds_content():
    from app import certificates

    kwargs = dict(
        certificate_id="CERT-x",
        batch_id="LOT-1",
        revision=1,
        plate={"type": "96", "rows": 8, "columns": 12},
        layout={"S1": "A1", "S2": "A2"},
        blocked_wells=["D6"],
        samples=[{"id": "S1", "group": "G"}, {"id": "S2", "group": "H"}],
        scan={"A1": "S1", "A2": "S2"},
        readings={"A1": 1.0, "A2": 2.0},
        thresholds={"max_spatial_effect": 0.5},
        seed=3,
        view_summary={"entries": 2, "digest": "ab"},
    )
    d1 = certificates._evidence_digest(**kwargs)
    d2 = certificates._evidence_digest(**kwargs)
    assert d1 == d2 and len(d1) == 64

    # Any field change (revision, reading, view summary, certificate id...)
    # changes the digest.
    changed = dict(kwargs, revision=2)
    assert certificates._evidence_digest(**changed) != d1
    changed = dict(kwargs, readings={"A1": 1.0, "A2": 2.0001})
    assert certificates._evidence_digest(**changed) != d1
    changed = dict(kwargs, view_summary={"entries": 2, "digest": "cd"})
    assert certificates._evidence_digest(**changed) != d1
    changed = dict(kwargs, certificate_id="CERT-y")
    assert certificates._evidence_digest(**changed) != d1


def test_reverify_digest_changes_only_when_evidence_changes():
    created = _certify()
    # Re-verification with identical evidence: digest should be the same
    # apart from the revision binding, i.e. still a fresh value per revision.
    r1 = client.post(
        "/api/v1/certificates/reverify",
        json=_reverify_payload(request_key="rv1", expected_revision=1),
    )
    assert r1.json()["digest"] != created["digest"]
    r1b = client.post(
        "/api/v1/certificates/reverify",
        json=_reverify_payload(request_key="rv1", expected_revision=1),
    )
    assert r1b.json()["digest"] == r1.json()["digest"]  # same content replay


def test_create_certificate_201_returns_id_status_revision_digest():
    body = _certify()
    assert body["status"] == "RELEASED"
    assert body["revision"] == 1
    assert body["certificate_id"].startswith("CERT-")
    assert len(body["digest"]) == 64
    int(body["digest"], 16)
    assert body["verdict"] == "RELEASED"
    assert body["gates"]["effect_within_limit"] is True
    assert body["scan_summary"]["matches_plan"] is True
    assert body["view_summary"] is None


def test_create_persists_layout_scan_readings_thresholds_in_one_transaction():
    body = _certify()
    from app import certificates

    store = certificates.get_store()
    cert = store.get_certificate("LOT-1")
    assert cert["status"] == "RELEASED" and cert["current_revision"] == 1
    snap = store.get_snapshot(body["certificate_id"], 1)
    assert json.loads(snap["layout_json"]) == {
        sid: well for sid, well in sorted(LAYOUT_4X4.items())
    }
    assert json.loads(snap["scan_json"]) == {
        w: s for w, s in sorted(
            ((w, s) for s, w in LAYOUT_4X4.items())
        )
    }
    assert json.loads(snap["readings_json"]) == {
        w: v for w, v in sorted(_readings().items())
    }
    assert json.loads(snap["thresholds_json"]) == {
        "max_spatial_effect": 10.0,
        "min_p_value": 0.001,
        "permutations": 100,
        "tolerance": 1e-6,
        "max_iterations": 100,
    }
    assert snap["evidence_digest"] == body["digest"]
    assert snap["view_summary_json"] is None

    events = store._conn.execute(
        "SELECT event, reason FROM certificate_events ORDER BY id"
    ).fetchall()
    assert [tuple(e) for e in events] == [("CREATED", None)]


def test_create_with_view_persists_only_the_view_summary():
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    body = _certify(view=view)
    assert body["view_summary"]["entries"] == 16
    assert len(body["view_summary"]["digest"]) == 64

    from app import certificates

    snap = certificates.get_store().get_snapshot(body["certificate_id"], 1)
    stored = snap["view_summary_json"]
    # No blind id, well-blinding or mapping content is persisted.
    for entry in view:
        assert entry["blind_id"] not in stored
    # And no share/package fields exist anywhere in the store.
    for table in ("certificates", "certificate_snapshots", "certificate_events",
                  "idempotency_keys"):
        rows = certificates.get_store()._conn.execute(
            f"SELECT * FROM {table}"
        ).fetchall()
        blob = json.dumps([dict(r) for r in rows], default=str)
        assert "share" not in blob.lower()
        assert "mapping_package" not in blob


def test_create_view_must_cover_exactly_layout_wells():
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items())[:-1], start=1)
    ]
    r = client.post("/api/v1/certificates", json=_create_payload(view=view))
    assert r.status_code == 422
    assert "CERT_VIEW_INVALID" in r.json()["infeasibility_categories"]


def test_create_view_rejects_duplicate_blind_ids_and_bad_wells():
    view = [
        {"well": well, "blind_id": "BL-same"}
        for _sid, well in sorted(LAYOUT_4X4.items())
    ]
    r = client.post("/api/v1/certificates", json=_create_payload(view=view))
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "CERT_VIEW_INVALID" in cats

    view_bad = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    view_bad[0]["well"] = "Z9"
    r = client.post("/api/v1/certificates", json=_create_payload(view=view_bad))
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


# ------------------------------------------------- NOT_RELEASED on create


def test_gate_failure_creates_no_certificate_but_is_replayable():
    payload = _create_payload(request_key="g1", max_spatial_effect=0.01)
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "NOT_RELEASED"
    assert "certificate_id" not in body
    assert body["reasons"] == ["SPATIAL_EFFECT_EXCEEDED"]
    assert body["batch_id"] == "LOT-1"

    from app import certificates

    assert certificates.get_store().get_certificate("LOT-1") is None

    # Same key + same content replays the stored verdict (no new computation).
    r2 = client.post("/api/v1/certificates", json=payload)
    assert r2.status_code == 200
    assert r2.json() == body

    # The batch can still be certified later with a different key.
    ok = _certify(request_key="g2")
    assert ok["revision"] == 1


def test_scan_mismatch_on_create_is_422_and_persists_nothing():
    payload = _create_payload(request_key="s1")
    del payload["scan"]["A1"]  # planned sample missing
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "SAMPLES_MISSING" in cats
    assert r.json()["misplaced_wells"] == [{"well": "A1", "expected": "gA1", "actual": None}]
    from app import certificates

    assert certificates.get_store().get_certificate("LOT-1") is None


# --------------------------------------------------------------- idempotency


def test_same_request_key_same_content_replays_verbatim():
    payload = _create_payload()
    r1 = client.post("/api/v1/certificates", json=payload)
    r2 = client.post("/api/v1/certificates", json=payload)
    assert r1.status_code == r2.status_code == 201
    assert r1.json() == r2.json()


def test_same_request_key_changed_content_is_409():
    _certify(request_key="dup")
    changed = _create_payload(request_key="dup")
    changed["readings"]["A1"] += 0.5
    r = client.post("/api/v1/certificates", json=changed)
    assert r.status_code == 409
    assert r.json()["conflict"] == "REQUEST_KEY_REUSED"


def test_request_key_cannot_be_reused_across_operations():
    body = _certify(request_key="shared")
    rv = _reverify_payload(request_key="shared", expected_revision=1)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 409
    assert r.json()["conflict"] == "REQUEST_KEY_REUSED"
    assert r.json()["certificate_id"] == body["certificate_id"]


def test_duplicate_batch_certification_is_409():
    _certify(request_key="first")
    r = client.post(
        "/api/v1/certificates", json=_create_payload(request_key="second")
    )
    assert r.status_code == 409
    assert r.json()["conflict"] == "BATCH_ALREADY_CERTIFIED"


# --------------------------------------------------------------- reverify ok


def test_reverify_passes_restores_frozen_params_and_increments_revision():
    created = _certify()
    # Submit only batch_id/expected_revision/scan/readings: thresholds, seed,
    # samples, layout and plate come from the frozen snapshot.
    rv = _reverify_payload(request_key="rv1", expected_revision=1)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "RELEASED"
    assert body["revision"] == 2
    assert body["certificate_id"] == created["certificate_id"]
    assert body["digest"] != created["digest"]
    assert body["thresholds"]["max_spatial_effect"] == 10.0
    assert body["plate"] == PLATE_4X4

    from app import certificates

    store = certificates.get_store()
    cert = store.get_certificate("LOT-1")
    assert cert["current_revision"] == 2 and cert["status"] == "RELEASED"
    # Both revisions are kept; the old snapshot is untouched.
    s1 = store.get_snapshot(created["certificate_id"], 1)
    s2 = store.get_snapshot(created["certificate_id"], 2)
    assert s1["evidence_digest"] == created["digest"]
    assert s2["evidence_digest"] == body["digest"]
    events = [
        tuple(e)
        for e in store._conn.execute(
            "SELECT event, reason FROM certificate_events ORDER BY id"
        ).fetchall()
    ]
    assert events == [("CREATED", None), ("REISSUED", None)]


def test_reverify_same_evidence_is_allowed_and_yields_new_revision():
    # The frozen gates use the same thresholds; identical evidence passes
    # again and simply appends another revision.
    _certify()
    for n in (2, 3):
        rv = _reverify_payload(
            request_key=f"rv{n}", expected_revision=n - 1
        )
        r = client.post("/api/v1/certificates/reverify", json=rv)
        assert r.status_code == 200, r.text
        assert r.json()["revision"] == n
        assert r.json()["status"] == "RELEASED"


def test_stale_expected_revision_is_409_and_does_not_overwrite():
    created = _certify()
    # advance to revision 2
    good = client.post(
        "/api/v1/certificates/reverify",
        json=_reverify_payload(request_key="rv1", expected_revision=1),
    )
    assert good.status_code == 200 and good.json()["revision"] == 2

    stale = client.post(
        "/api/v1/certificates/reverify",
        json=_reverify_payload(request_key="rv-old", expected_revision=1),
    )
    assert stale.status_code == 409
    assert stale.json()["conflict"] == "REVISION_MISMATCH"
    assert stale.json()["current_revision"] == 2

    from app import certificates

    # Revision 1 snapshot remains frozen with its original digest.
    s1 = certificates.get_store().get_snapshot(created["certificate_id"], 1)
    assert s1["evidence_digest"] == created["digest"]


def test_unknown_batch_reverify_is_404():
    rv = _reverify_payload(batch_id="GHOST")
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 404
    assert r.json()["conflict"] == "CERTIFICATE_NOT_FOUND"
    assert r.json()["batch_id"] == "GHOST"


# --------------------------------------------------------------- revocations


def test_reverify_scan_misplacement_revokes_with_first_reason_and_detail():
    created = _certify()
    rv = _reverify_payload(request_key="rv-bad", expected_revision=1)
    rv["scan"]["A1"], rv["scan"]["A2"] = rv["scan"]["A2"], rv["scan"]["A1"]
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["revision"] == 1  # no new revision is appended
    assert body["reason_category"] == "SCAN_MISMATCH"
    assert body["first_reason"] == "SCAN_MISMATCH"
    assert body["certificate_id"] == created["certificate_id"]
    cats = body["detail"]["infeasibility_categories"]
    assert "CLOSED_LOOP_MISPLACEMENT" in cats
    assert {"well": "A1", "expected": "gA1", "actual": "gB1"} in (
        body["detail"]["misplaced_wells"]
    )
    assert body["detail"]["cycles"] == [["A1", "A2"]]


def test_reverify_missing_sample_revokes():
    _certify()
    rv = _reverify_payload(request_key="rv-miss", expected_revision=1)
    del rv["scan"]["C3"]
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["first_reason"] == "SCAN_MISMATCH"
    assert "SAMPLES_MISSING" in body["detail"]["infeasibility_categories"]


def test_reverify_qc_gate_failure_revokes_with_specific_gate_first():
    # Certify with a generous effect limit, then re-verify with readings that
    # break the spatial gate. Bumping 3 of 4 wells in row A drives the row
    # effect well beyond the frozen limit (thresholds are restored, so the
    # caller cannot relax them).
    _certify(max_spatial_effect=3.0)
    bumped = _readings(row_bump_wells=("A1", "A2", "A3"), bump=10.0)
    rv = _reverify_payload(request_key="rv-qc", expected_revision=1, readings=bumped)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["reason_category"] == "SPATIAL_QC_FAILED"
    assert body["first_reason"] == "SPATIAL_EFFECT_EXCEEDED"
    assert body["detail"]["gates"]["effect_within_limit"] is False


def test_reverify_view_summary_change_revokes():
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    _certify(view=view)
    changed = [dict(e) for e in view]
    changed[0]["blind_id"] = "BL-ffffffffffff"
    rv = _reverify_payload(request_key="rv-view", expected_revision=1, view=changed)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVOKED"
    assert body["reason_category"] == "VIEW_CHANGED"
    assert body["first_reason"] == "VIEW_CHANGED"
    assert body["detail"]["frozen_view_summary"]["digest"] != (
        body["detail"]["submitted_view_summary"]["digest"]
    )


def test_reverify_view_added_when_none_was_bound_revokes():
    _certify()
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    rv = _reverify_payload(request_key="rv-add-view", expected_revision=1, view=view)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    assert r.json()["status"] == "REVOKED"
    assert r.json()["first_reason"] == "VIEW_CHANGED"


def test_reverify_view_omitted_when_bound_revokes():
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    _certify(view=view)
    rv = _reverify_payload(request_key="rv-no-view", expected_revision=1)  # no view
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    assert r.json()["status"] == "REVOKED"
    assert r.json()["first_reason"] == "VIEW_CHANGED"


def test_reverify_with_matching_view_passes_order_insensitively():
    view = [
        {"well": well, "blind_id": f"BL-{i:012d}"}
        for i, (_sid, well) in enumerate(sorted(LAYOUT_4X4.items()), start=1)
    ]
    created = _certify(view=view)
    rv = _reverify_payload(
        request_key="rv-view-ok", expected_revision=1, view=list(reversed(view))
    )
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "RELEASED" and body["revision"] == 2
    assert body["view_summary"]["digest"] == created["view_summary"]["digest"]


def test_revoked_certificate_cannot_be_restored():
    _certify()
    bad = _reverify_payload(request_key="rv-revoke", expected_revision=1)
    del bad["scan"]["A1"]
    revoked = client.post("/api/v1/certificates/reverify", json=bad)
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "REVOKED"

    # Even a perfectly passing re-verification afterwards is a 409.
    good = _reverify_payload(request_key="rv-restore", expected_revision=1)
    r = client.post("/api/v1/certificates/reverify", json=good)
    assert r.status_code == 409
    assert r.json()["conflict"] == "CERTIFICATE_REVOKED"

    # And a brand-new certificate for the same batch is also refused.
    r2 = client.post(
        "/api/v1/certificates",
        json=_create_payload(batch_id="LOT-1", request_key="create-after"),
    )
    assert r2.status_code == 409
    assert r2.json()["conflict"] == "CERTIFICATE_REVOKED"


def test_revocation_is_replayable_with_same_request_key():
    _certify()
    bad = _reverify_payload(request_key="rv-rep", expected_revision=1)
    del bad["scan"]["A1"]
    r1 = client.post("/api/v1/certificates/reverify", json=bad)
    r2 = client.post("/api/v1/certificates/reverify", json=bad)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert r2.json()["status"] == "REVOKED"


def test_structural_422_on_reverify_does_not_change_certificate():
    _certify()
    # A reading at a well that does not exist on the plate is malformed
    # evidence: 422, and the certificate stays live at revision 1.
    rv = _reverify_payload(request_key="rv-struct", expected_revision=1)
    rv["readings"]["Z9"] = 1.0
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 422

    good = _reverify_payload(request_key="rv-after-struct", expected_revision=1)
    r2 = client.post("/api/v1/certificates/reverify", json=good)
    assert r2.status_code == 200
    assert r2.json()["status"] == "RELEASED"
    assert r2.json()["revision"] == 2


def test_reverify_view_structure_error_is_422_without_revocation():
    _certify()
    rv = _reverify_payload(request_key="rv-view-bad", expected_revision=1)
    rv["view"] = [{"well": "A1", "blind_id": "BL-000000000001"},
                  {"well": "A1", "blind_id": "BL-000000000002"}]
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "CERT_VIEW_INVALID" in cats

    from app import certificates

    cert = certificates.get_store().get_certificate("LOT-1")
    assert cert["status"] == "RELEASED" and cert["current_revision"] == 1


# --------------------------------------------------------------- request shape


def test_missing_batch_id_is_request_validation_error():
    payload = _create_payload()
    del payload["batch_id"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_missing_request_key_is_request_validation_error():
    payload = _create_payload()
    del payload["request_key"]
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_reverify_expected_revision_must_be_positive():
    rv = _reverify_payload(expected_revision=0)
    r = client.post("/api/v1/certificates/reverify", json=rv)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_unknown_field_is_422():
    payload = _create_payload()
    payload["unexpected"] = 1
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_non_finite_reading_is_request_validation_error():
    payload = _create_payload()
    payload["readings"]["A1"] = float("nan")
    r = client.post(
        "/api/v1/certificates",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


# --------------------------------------------------------------- concurrency


def test_concurrent_same_key_requests_replay_instead_of_500():
    # FastAPI runs sync endpoints in a threadpool; a burst of identical
    # create requests must all succeed with the same stored certificate.
    import concurrent.futures

    payload = _create_payload(request_key="concurrent-key")

    def call():
        return client.post("/api/v1/certificates", json=payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _i: call(), range(8)))

    assert all(r.status_code == 201 for r in responses), [r.text for r in responses]
    ids = {r.json()["certificate_id"] for r in responses}
    assert ids == {responses[0].json()["certificate_id"]}
    assert len(ids) == 1

    from app import certificates

    cert = certificates.get_store().get_certificate("LOT-1")
    assert cert["current_revision"] == 1
    keys = certificates.get_store()._conn.execute(
        "SELECT COUNT(*) AS n FROM idempotency_keys"
    ).fetchone()
    assert keys["n"] == 1
