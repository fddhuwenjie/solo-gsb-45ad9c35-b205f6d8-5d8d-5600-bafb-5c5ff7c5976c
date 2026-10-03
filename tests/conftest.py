"""Shared pytest configuration.

Every test gets an isolated SQLite database for the certificate store: a new
temporary file is configured per test and the cached store singleton is reset.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _fresh_certificate_store():
    fd, path = tempfile.mkstemp(prefix="cert-test-", suffix=".db")
    os.close(fd)
    os.unlink(path)  # the store creates the file itself
    os.environ["CERT_DB_PATH"] = path
    from app import certificates

    certificates.reset_certificate_store()
    yield
    certificates.get_store().close()
    try:
        os.unlink(path)
    except OSError:
        pass
