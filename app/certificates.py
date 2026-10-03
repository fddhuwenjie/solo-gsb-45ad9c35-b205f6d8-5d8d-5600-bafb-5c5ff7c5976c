"""Release certificates for microplate batches that pass batch release.

A certificate freezes the evidence of a successful release verification
(``POST /api/v1/arrangements/batch-release``): the planned layout, the
agreed scan, the readings, every release threshold and — optionally — a
*summary* of the blinded operational view. All of that is written in a
single SQLite transaction together with the certificate row and the
idempotency record.

Re-verification (``POST .../certificates/reverify``) submits only a fresh
scan/readings pair; plate, target layout, blocked wells, samples, seed and
thresholds are restored from the frozen snapshot. Passing again appends a
new revision atomically (``revision`` increments, a new ``digest`` covers
the new evidence). Any scan disagreement, failed QC gate or changed view
summary marks the certificate ``REVOKED`` — irreversibly — and reports the
first failing reason.

Security: the store never receives or keeps key shares, mapping packages or
any plaintext blind correspondence. A submitted operational view is reduced
to a non-invertible summary (entry count plus a SHA-256 digest) before it is
persisted.

Idempotency: every request carries a client-chosen ``request_key``. The same
key with the same canonical content replays the stored response verbatim;
the same key with different content (or reused across the two operations) is
rejected with HTTP 409.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from .planner import InfeasibleArrangement, _resolve_dimensions
from .plates import parse_well, well_name
from .release import ReleaseInvalid, batch_release
from .schemas import (
    CERT_CONFLICT_BATCH_ALREADY_CERTIFIED,
    CERT_CONFLICT_CERTIFICATE_REVOKED,
    CERT_CONFLICT_REQUEST_KEY_REUSED,
    CERT_CONFLICT_REVISION_MISMATCH,
    CERT_REASON_SCAN_MISMATCH,
    CERT_REASON_SPATIAL_QC_FAILED,
    CERT_REASON_VIEW_CHANGED,
    CERT_STATUS_RELEASED,
    CERT_STATUS_REVOKED,
    RELEASE_REASON_P_VALUE_BELOW_THRESHOLD,
    RELEASE_REASON_P_VALUE_UNAVAILABLE,
    RELEASE_REASON_SPATIAL_EFFECT_EXCEEDED,
    VIOLATION_BLOCKED_WELL_OCCUPIED,
    VIOLATION_CERT_VIEW_INVALID,
    VIOLATION_CLOSED_LOOP_MISPLACEMENT,
    VIOLATION_SAMPLES_MISSING,
    VIOLATION_SCAN_LAYOUT_MISMATCH,
    VIOLATION_SCAN_SAMPLES_DUPLICATED,
    VIOLATION_SCAN_WELLS_DUPLICATED,
    VIOLATION_SPATIAL_NOT_CONVERGED,
    VIOLATION_TARGET_ON_BLOCKED,
    VIOLATION_TARGET_WELLS_DUPLICATED,
    VIOLATION_UNPLANNED_SAMPLES,
    VIOLATION_WELL_REFERENCE_INVALID,
    BatchReleaseRequest,
    CertificateCreateRequest,
    CertificateReverifyRequest,
    PlateSpec,
    SpatialSample,
)

OP_CREATE = "certificate.create"
OP_REVERIFY = "certificate.reverify"

_EVENT_CREATED = "CREATED"
_EVENT_REISSUED = "REISSUED"
_EVENT_REVOKED = "REVOKED"

_CERT_PREFIX = "CERT-"
_CERT_RANDOM_BYTES = 12  # 96-bit, collision re-checked inside the transaction

# Stage-1 (scan agreement) categories that revoke a certificate on
# re-verification. Everything else arriving in a ReleaseInvalid exception is
# a structural problem of the freshly submitted evidence (HTTP 422, no state
# change).
_SCAN_CATEGORIES = {
    VIOLATION_CLOSED_LOOP_MISPLACEMENT,
    VIOLATION_SCAN_LAYOUT_MISMATCH,
    VIOLATION_SAMPLES_MISSING,
    VIOLATION_UNPLANNED_SAMPLES,
    VIOLATION_SCAN_WELLS_DUPLICATED,
    VIOLATION_SCAN_SAMPLES_DUPLICATED,
    VIOLATION_TARGET_WELLS_DUPLICATED,
    VIOLATION_TARGET_ON_BLOCKED,
    VIOLATION_BLOCKED_WELL_OCCUPIED,
}

# Ordering of the gate reasons, so the reported first failure is stable.
_GATE_REASON_ORDER = {
    VIOLATION_SPATIAL_NOT_CONVERGED: 0,
    RELEASE_REASON_SPATIAL_EFFECT_EXCEEDED: 1,
    RELEASE_REASON_P_VALUE_UNAVAILABLE: 2,
    RELEASE_REASON_P_VALUE_BELOW_THRESHOLD: 3,
}


# ---------------------------------------------------------------- exceptions


class CertificateInvalid(Exception):
    """Malformed certificate request (HTTP 422); no certificate state changes."""

    def __init__(
        self,
        categories: list[str],
        messages: list[str],
        misplaced_wells: Optional[list[dict]] = None,
        cycles: Optional[list[list[str]]] = None,
    ):
        self.categories = categories
        self.messages = messages
        self.misplaced_wells = misplaced_wells or []
        self.cycles = cycles or []
        super().__init__("; ".join(messages) if messages else "invalid certificate request")


class CertificateConflict(Exception):
    """Request key / revision / lifecycle conflict (HTTP 409)."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        certificate_id: Optional[str] = None,
        status: Optional[str] = None,
        current_revision: Optional[int] = None,
    ):
        self.category = category
        self.message = message
        self.certificate_id = certificate_id
        self.status = status
        self.current_revision = current_revision
        super().__init__(message)


class CertificateMissing(Exception):
    """No certificate exists for the batch (HTTP 404)."""

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        super().__init__(f"no certificate exists for batch '{batch_id}'")


# --------------------------------------------------------------- primitives


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj) -> bytes:
    """Deterministic JSON encoding used for digests and replay comparisons."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _request_digest(req) -> str:
    """Content hash of a request excluding its idempotency key."""
    payload = req.model_dump(mode="json")
    payload.pop("request_key", None)
    return _sha256_hex(_canonical_json(payload))


def _canonical_view(view: list[dict], rows: int, cols: int) -> list[dict]:
    """Validate a submitted operational view; return it row-major canonical.

    Only public well/blind_id identifiers are involved. Well names must be
    legal and unique, and blind ids must be unique.
    """
    categories: list[str] = []
    messages: list[str] = []
    positions: list[tuple[int, int, str]] = []
    seen_wells: set[tuple[int, int]] = set()
    seen_blind: set[str] = set()
    for entry in view:
        well, blind_id = entry["well"], entry["blind_id"]
        try:
            pos = parse_well(well, rows, cols)
        except ValueError as exc:
            if VIOLATION_WELL_REFERENCE_INVALID not in categories:
                categories.append(VIOLATION_WELL_REFERENCE_INVALID)
            messages.append(str(exc))
            continue
        if pos in seen_wells:
            if VIOLATION_CERT_VIEW_INVALID not in categories:
                categories.append(VIOLATION_CERT_VIEW_INVALID)
            messages.append(f"well {well_name(*pos)} appears more than once in the view")
        if blind_id in seen_blind:
            if VIOLATION_CERT_VIEW_INVALID not in categories:
                categories.append(VIOLATION_CERT_VIEW_INVALID)
            messages.append(f"blind_id '{blind_id}' appears more than once in the view")
        seen_wells.add(pos)
        seen_blind.add(blind_id)
        positions.append((pos[0], pos[1], blind_id))

    if categories:
        raise CertificateInvalid(categories, messages)

    return [
        {"well": well_name(r, c), "blind_id": bid}
        for r, c, bid in sorted(positions, key=lambda x: (x[0], x[1]))
    ]


def _prepare_view(view, layout: dict[str, str], rows: int, cols: int) -> Optional[dict]:
    """Validate the view against the layout wells; build its stored summary.

    Returns ``None`` when no view was submitted. The summary is intentionally
    non-invertible: entry count plus a digest only — never the blind codes
    themselves, never any sample correspondence.
    """
    if view is None:
        return None
    canonical = _canonical_view([entry.model_dump() for entry in view], rows, cols)
    layout_wells = {parse_well(w, rows, cols) for w in layout.values()}
    view_wells = {parse_well(e["well"], rows, cols) for e in canonical}
    if view_wells != layout_wells:
        missing = sorted(well_name(*p) for p in layout_wells - view_wells)
        extra = sorted(well_name(*p) for p in view_wells - layout_wells)
        details = []
        if missing:
            details.append(f"view omits layout well(s): {', '.join(missing)}")
        if extra:
            details.append(f"view includes non-layout well(s): {', '.join(extra)}")
        raise CertificateInvalid(
            [VIOLATION_CERT_VIEW_INVALID],
            ["operational view must cover exactly the wells of the certified "
             "layout; " + "; ".join(details)],
        )
    return {
        "entries": len(canonical),
        "digest": _sha256_hex(_canonical_json(canonical)),
    }


def _evidence_digest(
    *,
    certificate_id: str,
    batch_id: str,
    revision: int,
    plate: dict,
    layout: dict[str, str],
    blocked_wells: list[str],
    samples: list[dict],
    scan: dict[str, str],
    readings: dict[str, Optional[float]],
    thresholds: dict,
    seed: int,
    view_summary: Optional[dict],
) -> str:
    """SHA-256 over the complete frozen evidence of one revision."""
    document = {
        "version": 1,
        "certificate_id": certificate_id,
        "batch_id": batch_id,
        "revision": revision,
        "plate": plate,
        "layout": {sid: layout[sid] for sid in sorted(layout)},
        "blocked_wells": sorted(blocked_wells),
        "samples": sorted(
            ({"id": s["id"], "group": s["group"]} for s in samples),
            key=lambda s: s["id"],
        ),
        "scan": {well: scan[well] for well in sorted(scan)},
        "readings": {well: readings[well] for well in sorted(readings)},
        "thresholds": thresholds,
        "seed": seed,
        "view_summary": view_summary,
    }
    return _sha256_hex(_canonical_json(document))


# ----------------------------------------------------------------- storage


_SCHEMA = """
CREATE TABLE IF NOT EXISTS certificates (
    certificate_id   TEXT PRIMARY KEY,
    batch_id         TEXT NOT NULL UNIQUE,
    status           TEXT NOT NULL,
    current_revision INTEGER NOT NULL,
    view_present     INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS certificate_snapshots (
    certificate_id  TEXT NOT NULL,
    revision        INTEGER NOT NULL,
    plate_json      TEXT NOT NULL,
    layout_json     TEXT NOT NULL,
    blocked_json    TEXT NOT NULL,
    samples_json    TEXT NOT NULL,
    scan_json       TEXT NOT NULL,
    readings_json   TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    view_summary_json TEXT,
    evidence_digest TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (certificate_id, revision),
    FOREIGN KEY (certificate_id) REFERENCES certificates(certificate_id)
);
CREATE TABLE IF NOT EXISTS certificate_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    certificate_id TEXT NOT NULL,
    revision       INTEGER NOT NULL,
    event          TEXT NOT NULL,
    reason         TEXT,
    detail_json    TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    request_key    TEXT PRIMARY KEY,
    operation      TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    certificate_id TEXT,
    status_code    INTEGER NOT NULL,
    response_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
"""


class CertificateStore:
    """SQLite-backed certificate store; all writes are atomic transactions."""

    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- reads ------------------------------------------------------------

    def lookup_key(self, request_key: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM idempotency_keys WHERE request_key = ?",
                (request_key,),
            ).fetchone()

    def get_certificate(self, batch_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM certificates WHERE batch_id = ?", (batch_id,)
            ).fetchone()

    def get_snapshot(self, certificate_id: str, revision: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM certificate_snapshots "
                "WHERE certificate_id = ? AND revision = ?",
                (certificate_id, revision),
            ).fetchone()

    # -- writes -----------------------------------------------------------

    def record_verdict_key(
        self,
        *,
        request_key: str,
        operation: str,
        request_digest: str,
        status_code: int,
        response: dict,
    ) -> None:
        """Persist an idempotency record for a verdict that made no certificate."""
        with self._lock, self._conn:
            self._insert_key(
                request_key, operation, request_digest, status_code, response, None, _now()
            )

    def insert_certificate(
        self,
        *,
        batch_id: str,
        parts: dict,
        view_present: bool,
        request_key: str,
        request_digest: str,
        make_response: Callable[[str, int, str], dict],
    ) -> tuple[str, int, str, dict]:
        """Create certificate + revision-1 snapshot + event + key atomically.

        The certificate id is minted inside the transaction; the evidence
        digest (which binds the id) and the stored response body are computed
        there too, so the persisted row and the returned body can never
        disagree.
        """
        with self._lock, self._conn:
            clash = self._conn.execute(
                "SELECT certificate_id, status FROM certificates WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if clash is not None:
                if clash["status"] == CERT_STATUS_REVOKED:
                    raise CertificateConflict(
                        CERT_CONFLICT_CERTIFICATE_REVOKED,
                        f"batch '{batch_id}' already has a revoked certificate; "
                        "revoked certificates cannot be restored",
                        certificate_id=clash["certificate_id"],
                        status=CERT_STATUS_REVOKED,
                    )
                raise CertificateConflict(
                    CERT_CONFLICT_BATCH_ALREADY_CERTIFIED,
                    f"batch '{batch_id}' already has a live certificate",
                    certificate_id=clash["certificate_id"],
                    status=CERT_STATUS_RELEASED,
                )

            certificate_id = self._new_certificate_id()
            revision = 1
            digest = _evidence_digest(
                certificate_id=certificate_id,
                batch_id=batch_id,
                revision=revision,
                plate=parts["plate"],
                layout=parts["layout"],
                blocked_wells=parts["blocked_wells"],
                samples=parts["samples"],
                scan=parts["scan"],
                readings=parts["readings"],
                thresholds=parts["thresholds"],
                seed=parts["seed"],
                view_summary=parts["view_summary"],
            )
            ts = _now()
            self._conn.execute(
                "INSERT INTO certificates "
                "(certificate_id, batch_id, status, current_revision, "
                " view_present, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    certificate_id,
                    batch_id,
                    CERT_STATUS_RELEASED,
                    revision,
                    1 if view_present else 0,
                    ts,
                    ts,
                ),
            )
            self._insert_snapshot(certificate_id, revision, ts, parts, digest)
            self._insert_event(certificate_id, revision, _EVENT_CREATED, None, None, ts)
            response = make_response(certificate_id, revision, digest)
            self._insert_key(
                request_key, OP_CREATE, request_digest, 201,
                response, certificate_id, ts,
            )
            return certificate_id, revision, digest, response

    def append_revision(
        self,
        *,
        certificate_id: str,
        revision: int,
        batch_id: str,
        parts: dict,
        request_key: str,
        request_digest: str,
        make_response: Callable[[str, int, str], dict],
    ) -> tuple[int, str, dict]:
        """Append revision n+1 + event + idempotency key in one transaction.

        The expected-revision check happens inside the transaction, so a
        frozen revision can never be overwritten.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT status, current_revision FROM certificates "
                "WHERE certificate_id = ?",
                (certificate_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - certificate is loaded beforehand
                raise CertificateMissing(certificate_id)
            if row["status"] == CERT_STATUS_REVOKED:
                raise CertificateConflict(
                    CERT_CONFLICT_CERTIFICATE_REVOKED,
                    "revoked certificates cannot be restored",
                    certificate_id=certificate_id,
                    status=CERT_STATUS_REVOKED,
                )
            if row["current_revision"] != revision:
                raise CertificateConflict(
                    CERT_CONFLICT_REVISION_MISMATCH,
                    f"expected_revision {revision} is not the current revision "
                    f"{row['current_revision']}; old revisions cannot be overwritten",
                    certificate_id=certificate_id,
                    status=CERT_STATUS_RELEASED,
                    current_revision=row["current_revision"],
                )
            new_revision = revision + 1
            digest = _evidence_digest(
                certificate_id=certificate_id,
                batch_id=batch_id,
                revision=new_revision,
                plate=parts["plate"],
                layout=parts["layout"],
                blocked_wells=parts["blocked_wells"],
                samples=parts["samples"],
                scan=parts["scan"],
                readings=parts["readings"],
                thresholds=parts["thresholds"],
                seed=parts["seed"],
                view_summary=parts["view_summary"],
            )
            ts = _now()
            self._insert_snapshot(certificate_id, new_revision, ts, parts, digest)
            self._conn.execute(
                "UPDATE certificates SET current_revision = ?, updated_at = ? "
                "WHERE certificate_id = ?",
                (new_revision, ts, certificate_id),
            )
            self._insert_event(
                certificate_id, new_revision, _EVENT_REISSUED, None, None, ts
            )
            response = make_response(certificate_id, new_revision, digest)
            self._insert_key(
                request_key, OP_REVERIFY, request_digest, 200,
                response, certificate_id, ts,
            )
            return new_revision, digest, response

    def revoke(
        self,
        *,
        certificate_id: str,
        revision: int,
        reason: str,
        detail: dict,
        request_key: str,
        request_digest: str,
        response: dict,
    ) -> None:
        """Flip the certificate to REVOKED, log event and key atomically.

        Revocation is terminal: the same transaction records the event and
        the idempotency record; no later request can change the status back.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT status FROM certificates WHERE certificate_id = ?",
                (certificate_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - loaded beforehand
                raise CertificateMissing(certificate_id)
            if row["status"] == CERT_STATUS_REVOKED:
                # The stored replay record for the original request remains
                # the canonical response; a new key cannot revive it.
                raise CertificateConflict(
                    CERT_CONFLICT_CERTIFICATE_REVOKED,
                    "revoked certificates cannot be restored",
                    certificate_id=certificate_id,
                    status=CERT_STATUS_REVOKED,
                )
            ts = _now()
            self._conn.execute(
                "UPDATE certificates SET status = ?, updated_at = ? "
                "WHERE certificate_id = ?",
                (CERT_STATUS_REVOKED, ts, certificate_id),
            )
            self._insert_event(
                certificate_id, revision, _EVENT_REVOKED, reason, detail, ts
            )
            self._insert_key(
                request_key, OP_REVERIFY, request_digest, 200,
                response, certificate_id, ts,
            )

    # -- internals (caller already holds the lock and transaction) --------

    def _new_certificate_id(self) -> str:
        while True:
            candidate = f"{_CERT_PREFIX}{secrets.token_hex(_CERT_RANDOM_BYTES)}"
            if self._conn.execute(
                "SELECT 1 FROM certificates WHERE certificate_id = ?", (candidate,)
            ).fetchone() is None:
                return candidate

    def _insert_snapshot(self, certificate_id, revision, ts, parts, digest) -> None:
        self._conn.execute(
            "INSERT INTO certificate_snapshots "
            "(certificate_id, revision, plate_json, layout_json, blocked_json, "
            " samples_json, scan_json, readings_json, thresholds_json, seed, "
            " view_summary_json, evidence_digest, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                certificate_id,
                revision,
                json.dumps(parts["plate"], ensure_ascii=False),
                json.dumps(parts["layout"], ensure_ascii=False),
                json.dumps(parts["blocked_wells"], ensure_ascii=False),
                json.dumps(parts["samples"], ensure_ascii=False),
                json.dumps(parts["scan"], ensure_ascii=False),
                json.dumps(parts["readings"], ensure_ascii=False),
                json.dumps(parts["thresholds"], ensure_ascii=False),
                parts["seed"],
                None
                if parts["view_summary"] is None
                else json.dumps(parts["view_summary"], ensure_ascii=False),
                digest,
                ts,
            ),
        )

    def _insert_event(self, certificate_id, revision, event, reason, detail, ts) -> None:
        self._conn.execute(
            "INSERT INTO certificate_events "
            "(certificate_id, revision, event, reason, detail_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                certificate_id,
                revision,
                event,
                reason,
                None if detail is None else json.dumps(detail, ensure_ascii=False),
                ts,
            ),
        )

    def _insert_key(
        self, request_key, operation, request_digest, status_code,
        response, certificate_id, ts,
    ) -> None:
        try:
            self._conn.execute(
                "INSERT INTO idempotency_keys "
                "(request_key, operation, request_digest, certificate_id, "
                " status_code, response_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    request_key,
                    operation,
                    request_digest,
                    certificate_id,
                    status_code,
                    json.dumps(response, ensure_ascii=False),
                    ts,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # Another process won the same-key race (the in-process flow
            # lock serialises threads): fail as a 409 rather than a 500.
            raise CertificateConflict(
                CERT_CONFLICT_REQUEST_KEY_REUSED,
                f"request_key '{request_key}' was already committed by a "
                "concurrent request; replay with the identical payload to "
                "obtain its stored response",
                certificate_id=certificate_id,
            ) from exc


# The store is a process-wide singleton keyed by its database path so tests
# can point CERT_DB_PATH at a temporary file.
_STORE: Optional[CertificateStore] = None
_STORE_PATH: Optional[str] = None
_STORE_GUARD = threading.Lock()

# A certificate flow is "idempotency check -> (verification) -> transactional
# write"; holding one process-wide lock across it makes the check-then-write
# atomic for the single-process uvicorn deployment (the deployment model in
# the shipped Dockerfile) so concurrent same-key requests can never race into
# a duplicate-key constraint violation.
_FLOW_LOCK = threading.RLock()


def get_store() -> CertificateStore:
    global _STORE, _STORE_PATH
    path = os.getenv("CERT_DB_PATH", os.path.join("data", "certificates.db"))
    if _STORE is None or _STORE_PATH != path:
        with _STORE_GUARD:
            if _STORE is None or _STORE_PATH != path:
                if _STORE is not None:
                    _STORE.close()
                _STORE = CertificateStore(path)
                _STORE_PATH = path
    return _STORE


def reset_certificate_store() -> CertificateStore:
    """Drop the cached store and reopen the configured path (tests)."""
    global _STORE, _STORE_PATH
    with _STORE_GUARD:
        if _STORE is not None:
            _STORE.close()
        _STORE = None
        _STORE_PATH = None
    return get_store()


# ------------------------------------------------------------- normalising


def _normalise_scan(scan: dict[str, str], rows: int, cols: int) -> dict[str, str]:
    return {
        well_name(*parse_well(raw, rows, cols)): sid
        for raw, sid in sorted(scan.items())
    }


def _normalise_readings(
    readings: dict[str, Optional[float]], rows: int, cols: int
) -> dict[str, Optional[float]]:
    return {
        well_name(*parse_well(raw, rows, cols)): value
        for raw, value in sorted(readings.items())
    }


def _normalise_blocked(blocked: list[str], rows: int, cols: int) -> list[str]:
    return sorted({well_name(*parse_well(w, rows, cols)) for w in blocked})


def _normalise_layout(layout: dict[str, str], rows: int, cols: int) -> dict[str, str]:
    return {
        sid: well_name(*parse_well(w, rows, cols))
        for sid, w in sorted(layout.items())
    }


def _normalise_samples(samples) -> list[dict]:
    out = [{"id": s.id, "group": s.group} for s in samples]
    out.sort(key=lambda s: s["id"])
    return out


def _thresholds_from_request(req) -> dict:
    return {
        "max_spatial_effect": req.max_spatial_effect,
        "min_p_value": req.min_p_value,
        "permutations": req.permutations,
        "tolerance": req.tolerance,
        "max_iterations": req.max_iterations,
    }


def _snapshot_parts(
    *,
    plate: dict,
    layout: dict[str, str],
    blocked_wells: list[str],
    samples: list[dict],
    scan: dict[str, str],
    readings: dict[str, Optional[float]],
    thresholds: dict,
    seed: int,
    view_summary: Optional[dict],
    rows: int,
    cols: int,
) -> dict:
    """Canonicalised evidence parts shared by the digest and the snapshot."""
    return {
        "plate": plate,
        "layout": _normalise_layout(layout, rows, cols),
        "blocked_wells": _normalise_blocked(blocked_wells, rows, cols),
        "samples": samples,
        "scan": _normalise_scan(scan, rows, cols),
        "readings": _normalise_readings(readings, rows, cols),
        "thresholds": thresholds,
        "seed": seed,
        "view_summary": view_summary,
    }


# ------------------------------------------------------------------ create


def _check_idempotency(
    store: CertificateStore, request_key: str, operation: str, request_digest: str
) -> Optional[tuple[int, dict]]:
    """Return the stored replay, or raise 409 on same-key/different-content."""
    row = store.lookup_key(request_key)
    if row is None:
        return None
    if row["operation"] == operation and row["request_digest"] == request_digest:
        return row["status_code"], json.loads(row["response_json"])
    suffix = (
        " for another operation"
        if row["operation"] != operation
        else " with different content"
    )
    raise CertificateConflict(
        CERT_CONFLICT_REQUEST_KEY_REUSED,
        f"request_key '{request_key}' was already used{suffix}; a request key "
        "cannot be reused with different content",
        certificate_id=row["certificate_id"],
    )


def create_certificate(req: CertificateCreateRequest) -> tuple[int, dict]:
    """Verify the release and, only on RELEASED, persist the certificate."""
    with _FLOW_LOCK:
        return _create_certificate_locked(req)


def _create_certificate_locked(req: CertificateCreateRequest) -> tuple[int, dict]:
    store = get_store()
    request_digest = _request_digest(req)
    replay = _check_idempotency(store, req.request_key, OP_CREATE, request_digest)
    if replay is not None:
        return replay

    try:
        rows, cols = _resolve_dimensions(req.plate)
    except InfeasibleArrangement as exc:
        raise CertificateInvalid(exc.categories, exc.messages) from exc

    # The view is structural input and is checked before any verdict work.
    view_summary = _prepare_view(req.view, req.target, rows, cols)

    # Reuse the release service unchanged: reconciliation rules (stage 1) and
    # spatial QC gates (stage 2).
    try:
        verdict = batch_release(req)
    except ReleaseInvalid as exc:
        raise CertificateInvalid(
            exc.categories,
            exc.messages,
            misplaced_wells=exc.misplaced_wells,
            cycles=exc.cycles,
        ) from exc

    plate = {"type": req.plate.type, "rows": rows, "columns": cols}

    if verdict["verdict"] != "RELEASED":
        # Gate failures are not malformed requests: no certificate is created,
        # but the verdict is replayable through request_key.
        body = dict(verdict)
        body["batch_id"] = req.batch_id
        store.record_verdict_key(
            request_key=req.request_key,
            operation=OP_CREATE,
            request_digest=request_digest,
            status_code=200,
            response=body,
        )
        return 200, body

    samples = _normalise_samples(req.samples)
    thresholds = _thresholds_from_request(req)
    parts = _snapshot_parts(
        plate=plate,
        layout=req.target,
        blocked_wells=req.blocked_wells,
        samples=samples,
        scan=req.scan,
        readings=req.readings,
        thresholds=thresholds,
        seed=req.seed,
        view_summary=view_summary,
        rows=rows,
        cols=cols,
    )

    def make_response(certificate_id: str, revision: int, digest: str) -> dict:
        return {
            "certificate_id": certificate_id,
            "batch_id": req.batch_id,
            "status": CERT_STATUS_RELEASED,
            "revision": revision,
            "digest": digest,
            "verdict": verdict["verdict"],
            "seed": req.seed,
            "plate": plate,
            "thresholds": thresholds,
            "gates": verdict["gates"],
            "scan_summary": verdict["scan_summary"],
            "view_summary": view_summary,
        }

    _cid, _rev, _digest, body = store.insert_certificate(
        batch_id=req.batch_id,
        parts=parts,
        view_present=view_summary is not None,
        request_key=req.request_key,
        request_digest=request_digest,
        make_response=make_response,
    )
    return 201, body


# ---------------------------------------------------------------- reverify


def _restored_request(snapshot_row: sqlite3.Row, scan, readings) -> BatchReleaseRequest:
    """Rebuild a BatchReleaseRequest from the frozen snapshot + new evidence."""
    return BatchReleaseRequest(
        plate=PlateSpec(**json.loads(snapshot_row["plate_json"])),
        target=json.loads(snapshot_row["layout_json"]),
        scan=scan,
        blocked_wells=json.loads(snapshot_row["blocked_json"]),
        samples=[SpatialSample(**s) for s in json.loads(snapshot_row["samples_json"])],
        readings=readings,
        seed=snapshot_row["seed"],
        **json.loads(snapshot_row["thresholds_json"]),
    )


def _is_scan_failure(categories: list[str], scan: dict[str, str], rows: int, cols: int) -> bool:
    """Classify a ReleaseInvalid on re-verification as scan-vs-structural."""
    if set(categories) & _SCAN_CATEGORIES:
        return True
    if VIOLATION_WELL_REFERENCE_INVALID in categories:
        # Target/blocked wells come from the already validated frozen
        # snapshot, so an unparseable well can only be in the fresh scan.
        for raw in scan:
            try:
                parse_well(raw, rows, cols)
            except ValueError:
                return True
    return False


def _first_gate_reason(reasons: list[str]) -> str:
    return sorted(reasons, key=lambda r: _GATE_REASON_ORDER.get(r, 99))[0]


def _revoke_response(
    *,
    certificate_id: str,
    batch_id: str,
    revision: int,
    reason_category: str,
    first_reason: str,
    message: str,
    detail: dict,
) -> dict:
    return {
        "certificate_id": certificate_id,
        "batch_id": batch_id,
        "status": CERT_STATUS_REVOKED,
        "revision": revision,
        "reason_category": reason_category,
        "first_reason": first_reason,
        "message": message,
        "detail": detail,
    }


def reverify_certificate(req: CertificateReverifyRequest) -> tuple[int, dict]:
    """Re-verify a certificate against fresh scan/readings evidence."""
    with _FLOW_LOCK:
        return _reverify_certificate_locked(req)


def _reverify_certificate_locked(req: CertificateReverifyRequest) -> tuple[int, dict]:
    store = get_store()
    request_digest = _request_digest(req)
    replay = _check_idempotency(store, req.request_key, OP_REVERIFY, request_digest)
    if replay is not None:
        return replay

    cert = store.get_certificate(req.batch_id)
    if cert is None:
        raise CertificateMissing(req.batch_id)
    certificate_id = cert["certificate_id"]

    if cert["status"] == CERT_STATUS_REVOKED:
        raise CertificateConflict(
            CERT_CONFLICT_CERTIFICATE_REVOKED,
            f"certificate for batch '{req.batch_id}' is REVOKED; revoked "
            "certificates cannot be restored",
            certificate_id=certificate_id,
            status=CERT_STATUS_REVOKED,
            current_revision=cert["current_revision"],
        )
    if cert["current_revision"] != req.expected_revision:
        raise CertificateConflict(
            CERT_CONFLICT_REVISION_MISMATCH,
            f"expected_revision {req.expected_revision} does not match the "
            f"current revision {cert['current_revision']}",
            certificate_id=certificate_id,
            status=CERT_STATUS_RELEASED,
            current_revision=cert["current_revision"],
        )

    revision = cert["current_revision"]
    snapshot_row = store.get_snapshot(certificate_id, revision)
    if snapshot_row is None:  # pragma: no cover - revisions are append-only/contiguous
        raise CertificateConflict(
            CERT_CONFLICT_REVISION_MISMATCH,
            f"snapshot for revision {revision} is missing; it cannot be re-verified",
            certificate_id=certificate_id,
            status=CERT_STATUS_RELEASED,
            current_revision=revision,
        )

    plate = json.loads(snapshot_row["plate_json"])
    rows, cols = plate["rows"], plate["columns"]
    frozen_view_summary = (
        None
        if snapshot_row["view_summary_json"] is None
        else json.loads(snapshot_row["view_summary_json"])
    )
    thresholds = json.loads(snapshot_row["thresholds_json"])
    frozen_samples = json.loads(snapshot_row["samples_json"])
    frozen_layout = json.loads(snapshot_row["layout_json"])
    frozen_blocked = json.loads(snapshot_row["blocked_json"])
    frozen_seed = snapshot_row["seed"]

    # A structurally invalid view is a malformed request (422), not a
    # verification failure: the certificate state is left untouched.
    submitted_view_summary: Optional[dict] = None
    if req.view is not None:
        canonical = _canonical_view(
            [entry.model_dump() for entry in req.view], rows, cols
        )
        submitted_view_summary = {
            "entries": len(canonical),
            "digest": _sha256_hex(_canonical_json(canonical)),
        }

    # Stage 1 + 2: everything but the fresh evidence comes from the snapshot.
    inner = _restored_request(snapshot_row, req.scan, req.readings)
    try:
        verdict = batch_release(inner)
    except ReleaseInvalid as exc:
        if _is_scan_failure(exc.categories, req.scan, rows, cols):
            reason = CERT_REASON_SCAN_MISMATCH
            message = (
                "re-verification scan does not agree with the frozen planned "
                f"layout: {'; '.join(exc.categories)}"
            )
            detail = {
                "infeasibility_categories": exc.categories,
                "messages": exc.messages,
                "misplaced_wells": exc.misplaced_wells,
                "cycles": exc.cycles,
            }
            body = _revoke_response(
                certificate_id=certificate_id,
                batch_id=req.batch_id,
                revision=revision,
                reason_category=CERT_REASON_SCAN_MISMATCH,
                first_reason=reason,
                message=message,
                detail=detail,
            )
            store.revoke(
                certificate_id=certificate_id,
                revision=revision,
                reason=reason,
                detail=detail,
                request_key=req.request_key,
                request_digest=request_digest,
                response=body,
            )
            return 200, body
        # Structural problems of the fresh evidence: 422, no state change.
        raise CertificateInvalid(
            exc.categories,
            exc.messages,
            misplaced_wells=exc.misplaced_wells,
            cycles=exc.cycles,
        ) from exc

    if verdict["verdict"] != "RELEASED":
        reason = _first_gate_reason(verdict["reasons"])
        message = (
            f"spatial QC re-verification failed at gate {reason}: "
            + "; ".join(verdict["messages"])
        )
        detail = {
            "reasons": verdict["reasons"],
            "messages": verdict["messages"],
            "gates": verdict["gates"],
        }
        body = _revoke_response(
            certificate_id=certificate_id,
            batch_id=req.batch_id,
            revision=revision,
            reason_category=CERT_REASON_SPATIAL_QC_FAILED,
            first_reason=reason,
            message=message,
            detail=detail,
        )
        store.revoke(
            certificate_id=certificate_id,
            revision=revision,
            reason=reason,
            detail=detail,
            request_key=req.request_key,
            request_digest=request_digest,
            response=body,
        )
        return 200, body

    # Stage 3: the view summary must be unchanged (its presence included).
    presence_differs = (frozen_view_summary is None) != (submitted_view_summary is None)
    content_differs = (
        frozen_view_summary is not None
        and submitted_view_summary is not None
        and frozen_view_summary != submitted_view_summary
    )
    if presence_differs or content_differs:
        if frozen_view_summary is None:
            message = "no blinded view was bound to the certificate but a view was submitted"
        elif submitted_view_summary is None:
            message = "the certificate binds a blinded view but none was submitted"
        else:
            message = (
                "the submitted operational view summary does not match the "
                "frozen view summary"
            )
        detail = {
            "frozen_view_summary": frozen_view_summary,
            "submitted_view_summary": submitted_view_summary,
        }
        body = _revoke_response(
            certificate_id=certificate_id,
            batch_id=req.batch_id,
            revision=revision,
            reason_category=CERT_REASON_VIEW_CHANGED,
            first_reason=CERT_REASON_VIEW_CHANGED,
            message=message,
            detail=detail,
        )
        store.revoke(
            certificate_id=certificate_id,
            revision=revision,
            reason=CERT_REASON_VIEW_CHANGED,
            detail=detail,
            request_key=req.request_key,
            request_digest=request_digest,
            response=body,
        )
        return 200, body

    # All gates passed: append the new revision atomically.
    parts = _snapshot_parts(
        plate=plate,
        layout=frozen_layout,
        blocked_wells=frozen_blocked,
        samples=frozen_samples,
        scan=req.scan,
        readings=req.readings,
        thresholds=thresholds,
        seed=frozen_seed,
        view_summary=frozen_view_summary,
        rows=rows,
        cols=cols,
    )

    def make_response(certificate_id: str, new_revision: int, digest: str) -> dict:
        return {
            "certificate_id": certificate_id,
            "batch_id": req.batch_id,
            "status": CERT_STATUS_RELEASED,
            "revision": new_revision,
            "digest": digest,
            "plate": plate,
            "thresholds": thresholds,
            "gates": verdict["gates"],
            "scan_summary": verdict["scan_summary"],
            "view_summary": frozen_view_summary,
        }

    # The append transaction recomputes the digest with batch_id bound and
    # verifies expected_revision inside the same transaction.
    _new_revision, _digest, body = store.append_revision(
        certificate_id=certificate_id,
        revision=revision,
        batch_id=req.batch_id,
        parts=parts,
        request_key=req.request_key,
        request_digest=request_digest,
        make_response=make_response,
    )
    return 200, body
