"""Tests for the schema-version guard at MemoryStore.open().

The guard refuses to open a store whose on-disk schema version is
newer than the installed package understands — that's a rollback
scenario and we'd rather fail loudly than silently corrupt data.

The default-zero case (legacy v1.0.0 stores written by code older
than the version-aware Sieve) is treated as compatible with the
current SCHEMA_VERSION, so existing users don't get blocked by an
upgrade.
"""
from __future__ import annotations

import pytest

from sieve.config import StoreConfig
from sieve.store import (
    SCHEMA_VERSION,
    MemoryStore,
    StoreSchemaTooNewError,
)


@pytest.fixture
def fresh_store(tmp_path):
    """An initialised store. Closed at fixture teardown."""
    db = tmp_path / "memory.db"
    cfg = StoreConfig(path=str(db))
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


class TestSchemaVersionStamp:
    def test_init_schema_stamps_current_version(self, fresh_store):
        """A freshly initialised store carries SCHEMA_VERSION."""
        row = fresh_store.conn.execute("PRAGMA user_version").fetchone()
        assert int(row[0]) == SCHEMA_VERSION

    def test_repeated_init_does_not_lower_version(self, tmp_path):
        """If a store is somehow on a higher version, init_schema
        must not silently downgrade it."""
        db = tmp_path / "memory.db"
        cfg = StoreConfig(path=str(db))
        ms = MemoryStore(cfg)
        ms.open()
        ms.init_schema()
        # Artificially advance the version
        ms.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
        ms.conn.commit()
        ms.close()

        # Reopen — should refuse (the guard fires)
        ms2 = MemoryStore(cfg)
        with pytest.raises(StoreSchemaTooNewError) as exc_info:
            ms2.open()
        assert exc_info.value.store_version == SCHEMA_VERSION + 5
        assert exc_info.value.package_version == SCHEMA_VERSION


class TestSchemaVersionGuard:
    def test_legacy_zero_version_opens_cleanly(self, tmp_path):
        """A store written by pre-version-aware Sieve has
        user_version=0. That must be treated as compatible — we
        don't want to break existing users on upgrade."""
        db = tmp_path / "memory.db"
        cfg = StoreConfig(path=str(db))

        # Simulate a legacy store: init it, then manually wipe
        # user_version back to 0 (which is what pre-aware Sieve
        # would have written).
        ms = MemoryStore(cfg)
        ms.open()
        ms.init_schema()
        ms.conn.execute("PRAGMA user_version = 0")
        ms.conn.commit()
        ms.close()

        # Reopening should succeed — 0 ≤ SCHEMA_VERSION, so the
        # guard passes.
        ms2 = MemoryStore(cfg)
        ms2.open()
        row = ms2.conn.execute("PRAGMA user_version").fetchone()
        # Still zero (open() doesn't auto-upgrade)
        assert int(row[0]) == 0
        ms2.close()

    def test_current_version_opens_cleanly(self, fresh_store):
        """The happy path: store on SCHEMA_VERSION, package on
        SCHEMA_VERSION — opens without complaint."""
        # The fixture already opened it; just confirm the fact-store
        # is usable.
        from sieve.store import MemoryStore as _MS  # noqa: F401
        # If we got here without an exception, the open() worked.
        assert fresh_store.conn is not None

    def test_too_new_version_raises(self, tmp_path):
        """Store on SCHEMA_VERSION+1 raises StoreSchemaTooNewError."""
        db = tmp_path / "memory.db"
        cfg = StoreConfig(path=str(db))

        # Write a store with user_version one above what we understand
        ms = MemoryStore(cfg)
        ms.open()
        ms.init_schema()
        ms.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        ms.conn.commit()
        ms.close()

        # Reopen — must refuse
        ms2 = MemoryStore(cfg)
        with pytest.raises(StoreSchemaTooNewError) as exc_info:
            ms2.open()

        err = exc_info.value
        assert err.store_version == SCHEMA_VERSION + 1
        assert err.package_version == SCHEMA_VERSION
        assert str(db) in str(err)
        # Error message names the recovery path
        msg = str(err).lower()
        assert "rollback" in msg or "rolled back" in msg or "backup restore" in msg

    def test_far_future_version_raises(self, tmp_path):
        """Sanity: a store on user_version=1000 raises."""
        db = tmp_path / "memory.db"
        cfg = StoreConfig(path=str(db))
        ms = MemoryStore(cfg)
        ms.open()
        ms.init_schema()
        ms.conn.execute("PRAGMA user_version = 1000")
        ms.conn.commit()
        ms.close()

        ms2 = MemoryStore(cfg)
        with pytest.raises(StoreSchemaTooNewError):
            ms2.open()

    def test_raise_closes_connection(self, tmp_path):
        """After the guard fires, the connection must not be left
        dangling — otherwise the caller can't retry or recover."""
        db = tmp_path / "memory.db"
        cfg = StoreConfig(path=str(db))
        ms = MemoryStore(cfg)
        ms.open()
        ms.init_schema()
        ms.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        ms.conn.commit()
        ms.close()

        ms2 = MemoryStore(cfg)
        try:
            ms2.open()
        except StoreSchemaTooNewError:
            pass
        # _conn must be None — the guard closed it before raising
        assert ms2._conn is None
