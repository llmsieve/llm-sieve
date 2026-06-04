"""Key management for `sieve key show/rotate/export/import`.

SQLCipher re-encryption flow (``rotate_key``):
  1. Verify the old key actually opens the DB.
  2. Snapshot the keyfile for rollback (``.sieve_key.bak``).
  3. Open the DB with the old key, run ``PRAGMA rekey`` with the new key.
     SQLCipher rewrites every page in-place inside a transaction.
  4. Close the connection; write the new key to the keyfile at 0600.
  5. On any failure between step 3 and 5, restore the backup keyfile.

All helpers here are pure (no CLI, no prompting). The click wrappers in
cli.py handle user interaction.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def fingerprint(key: str) -> str:
    """SHA-256 of the key; return the first 16 hex chars (display only)."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def keyfile_for(db_path: Path) -> Path:
    return db_path.parent / ".sieve_key"


def verify_key(db_path: Path, key: str) -> bool:
    """Return True if ``key`` successfully opens ``db_path``."""
    if not db_path.exists():
        return False
    try:
        import sqlcipher3
        conn = sqlcipher3.connect(str(db_path))
        conn.execute(f"PRAGMA key='{key}'")
        conn.execute("SELECT count(*) FROM sqlite_master")
        conn.close()
        return True
    except Exception:
        return False


def generate_key() -> str:
    """Fresh 64-char hex — matches the default format written at install."""
    return os.urandom(32).hex()


def rotate_key(db_path: Path, old_key: str, new_key: str) -> None:
    """Re-encrypt ``db_path`` with ``new_key`` and update the keyfile.

    Raises ValueError if ``old_key`` doesn't open the DB.
    """
    if not verify_key(db_path, old_key):
        raise ValueError("current key does not open the store")

    kf = keyfile_for(db_path)
    kf_bak = kf.with_suffix(kf.suffix + ".bak")

    # Backup keyfile (if it exists) so we can roll back on failure.
    if kf.exists():
        shutil.copy2(kf, kf_bak)

    try:
        import sqlcipher3
        conn = sqlcipher3.connect(str(db_path))
        # Open with the old key then issue rekey.
        conn.execute(f"PRAGMA key='{old_key}'")
        # sqlite-vec extension is NOT required for rekey; skip loading it.
        # Also commit any open WAL by checkpointing first so rekey catches
        # all pages.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute(f"PRAGMA rekey='{new_key}'")
        conn.close()

        # Atomic keyfile swap: write-temp + rename.
        kf.parent.mkdir(parents=True, exist_ok=True)
        tmp = kf.with_suffix(kf.suffix + ".tmp")
        tmp.write_text(new_key)
        tmp.chmod(0o600)
        os.replace(tmp, kf)

    except Exception:
        # Roll back on any failure during rekey / keyfile write.
        if kf_bak.exists():
            shutil.copy2(kf_bak, kf)
        raise
    finally:
        if kf_bak.exists():
            kf_bak.unlink()


def import_key(db_path: Path, source_keyfile: Path) -> None:
    """Copy ``source_keyfile`` → the canonical keyfile location after
    verifying it opens the DB.

    Raises ValueError if the key doesn't work.
    """
    if not source_keyfile.exists():
        raise FileNotFoundError(source_keyfile)

    candidate = source_keyfile.read_text().strip()
    if not verify_key(db_path, candidate):
        raise ValueError("provided key does not open the current store")

    target = keyfile_for(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate)
    target.chmod(0o600)
