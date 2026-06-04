"""Tests for `sieve update` — the on-request PyPI version check."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from sieve.cli_update import (
    _detect_install_path,
    _format_release_date,
    _parse_version,
    run_update_check,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _pypi_response(latest_version: str, *, upload_time: str | None = None):
    """A minimal PyPI JSON-API-shaped dict for the given latest version."""
    payload = {
        "info": {"version": latest_version},
        "releases": {
            latest_version: [
                {"upload_time_iso_8601": upload_time or "2026-09-14T10:33:21.123456Z"}
            ]
        },
    }
    return payload


class _FakeResponse:
    def __init__(self, json_data, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )

    def json(self):
        if self._json is None:
            raise ValueError("not JSON")
        return self._json


# ── unit-level: helpers ──────────────────────────────────────────────────


class TestParseVersion:
    def test_packaging_present_returns_comparable(self):
        a = _parse_version("1.0.0")
        b = _parse_version("1.1.0")
        assert a < b

    def test_handles_prerelease_ordering(self):
        # 1.0.0 < 1.1.0rc1 < 1.1.0
        assert _parse_version("1.0.0") < _parse_version("1.1.0rc1")
        assert _parse_version("1.1.0rc1") < _parse_version("1.1.0")

    def test_handles_dotted_garbage_gracefully(self):
        # Not a SemVer string, but we should still return something
        # comparable rather than raise.
        v = _parse_version("not-a-version")
        # No assertion on ordering; just that the call returns.
        assert v is not None


class TestFormatReleaseDate:
    def test_isoformat_truncates_to_yyyy_mm_dd(self):
        assert _format_release_date("2026-09-14T10:33:21.123456Z") == "2026-09-14"

    def test_none_returns_empty(self):
        assert _format_release_date(None) == ""

    def test_empty_string_returns_empty(self):
        assert _format_release_date("") == ""


class TestDetectInstallPath:
    def test_returns_one_of_the_three_canonical_values(self):
        result = _detect_install_path()
        assert result in {"pipx", "pip", "unknown"}


# ── integration-ish: run_update_check end-to-end with mocked HTTP ──────


def _patch_httpx(monkeypatch, response_factory):
    """Patch httpx.get to call response_factory(url, timeout) -> response."""
    import httpx
    monkeypatch.setattr(httpx, "get", response_factory)


class TestRunUpdateCheck:
    def test_on_latest_prints_no_upgrade_needed(self, monkeypatch, capsys):
        # Local 1.0.0, PyPI says 1.0.0 — user is current.
        monkeypatch.setattr("sieve.__version__", "1.0.0")
        _patch_httpx(
            monkeypatch,
            lambda url, timeout: _FakeResponse(_pypi_response("1.0.0")),
        )
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "on the latest version" in out.lower()
        assert "1.0.0" in out

    def test_newer_version_prints_upgrade_command(self, monkeypatch, capsys):
        # Local 1.0.0, PyPI says 1.1.0 — newer available.
        monkeypatch.setattr("sieve.__version__", "1.0.0")
        _patch_httpx(
            monkeypatch,
            lambda url, timeout: _FakeResponse(
                _pypi_response("1.1.0", upload_time="2026-09-14T10:33:21.123456Z")
            ),
        )
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "newer version" in out.lower()
        assert "1.1.0" in out
        # Both upgrade commands present
        assert "pipx upgrade llm-sieve" in out
        assert "pip install --upgrade llm-sieve" in out
        # Release-notes URL present
        assert "releases/tag/v1.1.0" in out
        # Backup tip present
        assert "sieve backup create" in out.lower() or "backup create" in out

    def test_timeout_is_non_fatal(self, monkeypatch, capsys):
        import httpx
        def raise_timeout(url, timeout):
            raise httpx.TimeoutException("simulated timeout")
        monkeypatch.setattr(httpx, "get", raise_timeout)
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "timed out" in out.lower() or "couldn't" in out.lower()
        # User's local Sieve is unaffected — message says so
        assert "unaffected" in out.lower() or "local sieve" in out.lower()

    def test_404_is_non_fatal(self, monkeypatch, capsys):
        import httpx
        def raise_http_error(url, timeout):
            # httpx HTTPStatusError needs a Request — fake it minimally
            req = httpx.Request("GET", url)
            resp = httpx.Response(404, request=req)
            raise httpx.HTTPStatusError("404", request=req, response=resp)
        monkeypatch.setattr(httpx, "get", raise_http_error)
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "couldn't" in out.lower() or "404" in out

    def test_invalid_json_returns_error(self, monkeypatch, capsys):
        _patch_httpx(
            monkeypatch,
            # raise_for_status passes, .json() raises
            lambda url, timeout: _FakeResponse(None),
        )
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "invalid" in out.lower() or "pypi" in out.lower()

    def test_missing_version_field_returns_error(self, monkeypatch, capsys):
        _patch_httpx(
            monkeypatch,
            lambda url, timeout: _FakeResponse({"info": {}}),
        )
        exit_code = run_update_check()
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "no version" in out.lower() or "pypi" in out.lower()


# ── command-level: `sieve update` CLI ─────────────────────────────────


class TestUpdateCommand:
    def test_command_is_registered(self):
        from sieve.cli import cli
        result = CliRunner().invoke(cli, ["update", "--help"])
        assert result.exit_code == 0
        assert "update" in result.output.lower()
        # Help text mentions the privacy commitment
        assert "auto" in result.output.lower() or "on-request" in result.output.lower()

    def test_no_telemetry_on_help(self):
        """`--help` must not hit the network. Spot-check by running it
        with no network mocking — should still succeed instantly."""
        from sieve.cli import cli
        import time
        t0 = time.monotonic()
        result = CliRunner().invoke(cli, ["update", "--help"])
        elapsed = time.monotonic() - t0
        assert result.exit_code == 0
        # Help should be near-instant; if this takes >2s, something
        # is making a network call it shouldn't be.
        assert elapsed < 2.0, f"update --help took {elapsed:.2f}s — possible network call"
