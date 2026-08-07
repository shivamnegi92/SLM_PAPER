"""Pytest config: mark and auto-skip network-dependent integration tests.

Uses urllib (which respects HTTP_PROXY/HTTPS_PROXY env vars) rather than raw
sockets, since some networks only route through an HTTP(S) proxy. No
proxy-specific hostnames are hardcoded here -- set the standard env vars in
your shell if you need one.
"""
from __future__ import annotations

import urllib.request

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: requires network access to a public dataset hub"
    )


def _network_available(url: str = "https://huggingface.co", timeout: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "slmpaper-ci"})
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def network_available() -> bool:
    return _network_available()
