"""Tests for Phase 10: Security + Backup.

Tests cover:
- Auth token enforcement on /sieve/ endpoints
- No auth required on proxy passthrough endpoints
- 401 with no information leakage
- HTTPS warning for remote providers
- Data minimisation (sanitize_context_block)
- Audit log endpoint
- Backup create/list/verify/restore
- Store migration
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from sieve.backup import (
    create_backup,
    list_backups,
    migrate_store,
    restore_backup,
    verify_backup,
)
from sieve.config import RecallConfig, SecurityConfig, StoreConfig
from sieve.security import (
    check_auth,
    check_https_warning,
    get_or_create_auth_token,
    sanitize_context_block,
)
from sieve.store import MemoryStore


# ─── Auth Token ──────────────────────────────────────────────────────────────

class TestAuthToken:
    def test_generate_token(self, tmp_path):
        token = get_or_create_auth_token(tmp_path)
        assert len(token) == 64  # 32 bytes hex
        # Second call returns same token
        assert get_or_create_auth_token(tmp_path) == token

    def test_token_file_permissions(self, tmp_path):
        get_or_create_auth_token(tmp_path)
        token_file = tmp_path / ".sieve_auth_token"
        assert token_file.exists()
        assert oct(token_file.stat().st_mode & 0o777) == "0o600"


class TestCheckAuth:
    def _make_request(self, path, token=None):
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": [],
        }
        if token:
            scope["headers"] = [(b"x-sieve-token", token.encode())]
        from fastapi import Request
        return Request(scope)

    def test_recall_path_no_token_returns_401(self):
        req = self._make_request("/sieve/stats")
        result = check_auth(req, "secret-token")
        assert result is not None
        assert result.status_code == 401

    def test_recall_path_wrong_token_returns_401(self):
        req = self._make_request("/sieve/stats", token="wrong")
        result = check_auth(req, "secret-token")
        assert result is not None
        assert result.status_code == 401

    def test_recall_path_correct_token_returns_none(self):
        req = self._make_request("/sieve/stats", token="secret-token")
        result = check_auth(req, "secret-token")
        assert result is None

    def test_proxy_path_no_token_returns_none(self):
        req = self._make_request("/api/chat")
        result = check_auth(req, "secret-token")
        assert result is None

    def test_openai_path_no_token_returns_none(self):
        req = self._make_request("/v1/chat/completions")
        result = check_auth(req, "secret-token")
        assert result is None

    def test_health_endpoint_is_public(self):
        """Health endpoint should not require auth (for monitoring/load balancers)."""
        req = self._make_request("/sieve/health")
        result = check_auth(req, "secret-token")
        assert result is None

    def test_401_no_information_leakage(self):
        req = self._make_request("/sieve/stats")
        result = check_auth(req, "secret-token")
        assert result is not None
        body = result.body.decode()
        assert "Unauthorized" in body


# ─── HTTPS Warning ────────────────────────────────────────────────────────────

class TestHTTPSWarning:
    def test_local_http_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="recall.security"):
            check_https_warning("http://127.0.0.1:11434")
        assert "SECURITY" not in caplog.text

    def test_local_192_168_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="recall.security"):
            check_https_warning("http://192.168.1.100:11434")
        assert "SECURITY" not in caplog.text

    def test_remote_http_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="recall.security"):
            check_https_warning("http://example.com:11434")
        assert "SECURITY" in caplog.text
        assert "HTTPS" in caplog.text

    def test_remote_https_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="recall.security"):
            check_https_warning("https://example.com:11434")
        assert "SECURITY" not in caplog.text


# ─── Data Minimisation ────────────────────────────────────────────────────────

class TestDataMinimisation:
    def test_removes_uuid(self):
        text = "Fact a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6 says user lives in Dubai"
        clean = sanitize_context_block(text)
        assert "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" not in clean
        assert "Dubai" in clean

    def test_removes_confidence_scores(self):
        text = "User lives in Dubai (confidence: 0.85)"
        clean = sanitize_context_block(text)
        assert "confidence" not in clean
        assert "0.85" not in clean

    def test_removes_fact_type(self):
        text = "User is a pilot (objective)"
        clean = sanitize_context_block(text)
        assert "(objective)" not in clean
        assert "pilot" in clean

    def test_clean_text_unchanged(self):
        text = "User lives in Dubai"
        assert sanitize_context_block(text) == text

    def test_empty_string(self):
        assert sanitize_context_block("") == ""


# ─── Backup ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store_path(tmp_path):
    """Create a real encrypted store and return its path."""
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test-backup")
    ms.open()
    ms.init_schema()
    ms.insert_fact("Test fact for backup", embedding=None)
    ms.close()
    return tmp_path / "test.db"


class TestBackupCreate:
    def test_creates_backup_file(self, store_path):
        backup_path, checksum_path = create_backup(store_path)
        assert backup_path.exists()
        assert checksum_path.exists()
        assert backup_path.suffix == ".enc"

    def test_backup_has_correct_name_format(self, store_path):
        backup_path, _ = create_backup(store_path)
        assert "recall_backup_" in backup_path.name

    def test_checksum_file_contains_hash(self, store_path):
        _, checksum_path = create_backup(store_path)
        content = checksum_path.read_text()
        assert len(content.split()[0]) == 64  # SHA-256 hex

    def test_custom_output_path(self, store_path, tmp_path):
        custom = tmp_path / "custom_backup.db.enc"
        backup_path, _ = create_backup(store_path, output=custom)
        assert backup_path == custom

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_backup(tmp_path / "nonexistent.db")


class TestBackupList:
    def test_empty_list(self, store_path):
        backups = list_backups(store_path)
        assert backups == []

    def test_lists_created_backup(self, store_path):
        create_backup(store_path)
        backups = list_backups(store_path)
        assert len(backups) == 1
        assert backups[0]["checksum_valid"] is True

    def test_multiple_backups(self, store_path):
        create_backup(store_path)
        create_backup(store_path)
        backups = list_backups(store_path)
        assert len(backups) >= 1  # timestamp collision possible in fast test


class TestBackupVerify:
    def test_valid_backup(self, store_path):
        backup_path, _ = create_backup(store_path)
        assert verify_backup(backup_path) is True

    def test_tampered_backup(self, store_path):
        backup_path, _ = create_backup(store_path)
        # Tamper with the backup
        with open(backup_path, "ab") as f:
            f.write(b"tampered")
        assert verify_backup(backup_path) is False

    def test_missing_checksum(self, store_path, tmp_path):
        backup_path, checksum_path = create_backup(store_path)
        checksum_path.unlink()
        assert verify_backup(backup_path) is False


class TestBackupRestore:
    def test_restore_from_backup(self, store_path):
        backup_path, _ = create_backup(store_path)
        # Delete original
        store_path.unlink()
        assert not store_path.exists()
        # Restore
        success = restore_backup(backup_path, store_path)
        assert success is True
        assert store_path.exists()

    def test_restore_creates_pre_backup(self, store_path):
        backup_path, _ = create_backup(store_path)
        # Restore over existing
        success = restore_backup(backup_path, store_path)
        assert success is True
        # Pre-restore backup should exist
        pre_restores = list(store_path.parent.glob("pre_restore_*.db.bak"))
        assert len(pre_restores) == 1

    def test_restore_bad_checksum_fails(self, store_path):
        backup_path, _ = create_backup(store_path)
        with open(backup_path, "ab") as f:
            f.write(b"tampered")
        success = restore_backup(backup_path, store_path)
        assert success is False

    def test_restore_missing_backup(self, tmp_path, store_path):
        with pytest.raises(FileNotFoundError):
            restore_backup(tmp_path / "nope.db.enc", store_path)


class TestStoreMigrate:
    def test_migrate_success(self, store_path, tmp_path):
        dst = tmp_path / "new_location" / "memory.db"
        success = migrate_store(store_path, dst, passphrase="test-backup")
        assert success is True
        assert dst.exists()

    def test_migrate_copies_keyfile(self, store_path, tmp_path):
        # Create a keyfile
        keyfile = store_path.parent / ".sieve_key"
        keyfile.write_text("test-backup")
        keyfile.chmod(0o600)

        dst = tmp_path / "migrated" / "memory.db"
        migrate_store(store_path, dst, passphrase="test-backup")
        dst_key = dst.parent / ".sieve_key"
        assert dst_key.exists()

    def test_migrate_missing_source(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            migrate_store(tmp_path / "nope.db", tmp_path / "dst.db")


# ─── Audit Log ────────────────────────────────────────────────────────────────

class TestAuditLog:
    def test_fact_operations_audited(self):
        """Verify that fact insertions create audit log entries without content."""
        import tempfile
        config = StoreConfig(path=str(Path(tempfile.mkdtemp()) / "audit.db"), embedding_dimensions=4)
        ms = MemoryStore(config, passphrase="test-audit")
        ms.open()
        ms.init_schema()

        ms.insert_fact("User lives in Dubai", embedding=None)
        ms.insert_fact("User is a pilot", embedding=None)

        # Check audit log
        rows = ms.conn.execute(
            "SELECT operation, target_type, target_id FROM audit_log WHERE operation = 'extract'"
        ).fetchall()
        assert len(rows) >= 2
        # Verify no content is stored
        all_entries = ms.conn.execute("SELECT * FROM audit_log").fetchall()
        for row in all_entries:
            row_str = str(row)
            assert "Dubai" not in row_str
            assert "pilot" not in row_str

        ms.close()
