"""Backup — encrypted backup creation, listing, restoration, and store migration.

Backups are always encrypted (same SQLCipher key as the source store).
Timestamped as recall_backup_YYYY-MM-DDTHHMMSS.db.enc with SHA-256 checksum.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("recall.backup")

_BACKUP_DIR_NAME = "backups"


def _backup_dir(db_path: Path) -> Path:
    """Get backup directory next to the database."""
    d = db_path.parent / _BACKUP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_backup(db_path: Path, output: Path | None = None) -> tuple[Path, Path]:
    """Create an encrypted backup copy of the database.

    In SQLCipher's default WAL mode, recent commits live in a `-wal` sidecar
    until checkpointed. A backup taken while the store is still open would
    miss those commits if we only copied the main `.db` file. To keep the
    backup self-consistent we also copy any `-wal` and `-shm` sidecar files
    next to the main backup using the same stem + suffix; `restore_backup`
    puts them back next to the restored database.

    Returns (backup_path, checksum_path). Only the main `.db.enc` file has
    a checksum; sidecars are short-lived and carry no integrity guarantee.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    ts = _timestamp()
    if output is None:
        backup_dir = _backup_dir(db_path)
        output = backup_dir / f"recall_backup_{ts}.db.enc"

    # Copy the main encrypted file (it's already encrypted by SQLCipher).
    shutil.copy2(str(db_path), str(output))

    # Copy any WAL sidecars alongside the backup so restore is self-consistent.
    wal_src = db_path.with_name(db_path.name + "-wal")
    shm_src = db_path.with_name(db_path.name + "-shm")
    if wal_src.exists():
        shutil.copy2(str(wal_src), str(output) + "-wal")
    if shm_src.exists():
        shutil.copy2(str(shm_src), str(output) + "-shm")

    # Generate SHA-256 checksum of the main file
    checksum = _sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n")

    logger.info("Backup created: %s (%s)", output, checksum[:16])
    return output, checksum_path


def list_backups(db_path: Path) -> list[dict]:
    """List available backups with timestamps, sizes, and checksum status."""
    backup_dir = _backup_dir(db_path)
    results = []

    for f in sorted(backup_dir.glob("recall_backup_*.db.enc")):
        checksum_file = f.with_suffix(f.suffix + ".sha256")
        has_checksum = checksum_file.exists()
        checksum_valid = False

        if has_checksum:
            expected = checksum_file.read_text().split()[0]
            actual = _sha256(f)
            checksum_valid = expected == actual

        results.append({
            "id": f.stem,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "timestamp": f.stem.replace("recall_backup_", ""),
            "has_checksum": has_checksum,
            "checksum_valid": checksum_valid,
        })

    return results


def verify_backup(backup_path: Path) -> bool:
    """Verify a backup's SHA-256 checksum."""
    checksum_file = backup_path.with_suffix(backup_path.suffix + ".sha256")
    if not checksum_file.exists():
        logger.warning("No checksum file for %s", backup_path)
        return False

    expected = checksum_file.read_text().split()[0]
    actual = _sha256(backup_path)
    return expected == actual


def restore_backup(backup_path: Path, db_path: Path) -> bool:
    """Restore a database from a backup.

    Verifies checksum before restoring. Also restores any `-wal` / `-shm`
    sidecars that were captured alongside the backup. Any stale sidecars at
    the target path are removed first so they can't corrupt the restored DB.

    Returns True on success.
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    # Verify checksum first
    if not verify_backup(backup_path):
        logger.error("Backup checksum verification failed for %s", backup_path)
        return False

    # Create a backup of the current database before overwriting
    if db_path.exists():
        pre_restore = db_path.parent / f"pre_restore_{_timestamp()}.db.bak"
        shutil.copy2(str(db_path), str(pre_restore))
        logger.info("Pre-restore backup: %s", pre_restore)

    # Remove stale WAL sidecars at the target — they belong to the old DB
    # and would be inconsistent with the restored main file.
    stale_wal = db_path.with_name(db_path.name + "-wal")
    stale_shm = db_path.with_name(db_path.name + "-shm")
    for stale in (stale_wal, stale_shm):
        if stale.exists():
            stale.unlink()

    shutil.copy2(str(backup_path), str(db_path))

    # Restore sidecars if they were captured alongside the backup
    wal_src = Path(str(backup_path) + "-wal")
    shm_src = Path(str(backup_path) + "-shm")
    if wal_src.exists():
        shutil.copy2(str(wal_src), str(db_path) + "-wal")
    if shm_src.exists():
        shutil.copy2(str(shm_src), str(db_path) + "-shm")

    logger.info("Restored from %s to %s", backup_path, db_path)
    return True


def migrate_store(
    src_path: Path,
    dst_path: Path,
    passphrase: str | None = None,
) -> bool:
    """Copy the store to a new location, verify integrity.

    Returns True on success.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source store not found: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(dst_path))

    # Copy the keyfile too
    src_key = src_path.parent / ".sieve_key"
    if src_key.exists():
        dst_key = dst_path.parent / ".sieve_key"
        shutil.copy2(str(src_key), str(dst_key))
        dst_key.chmod(0o600)

    # Verify integrity by opening
    try:
        import sqlcipher3
        import sqlite_vec
        from sieve.store import get_or_create_passphrase

        pp = passphrase or get_or_create_passphrase(dst_path)
        conn = sqlcipher3.connect(str(dst_path))
        conn.execute(f"PRAGMA key='{pp}'")
        conn.execute("SELECT count(*) FROM facts")
        conn.close()
        logger.info("Migration verified: %s → %s", src_path, dst_path)
        return True
    except Exception as exc:
        logger.error("Migration verification failed: %s", exc)
        # Clean up failed migration
        if dst_path.exists():
            dst_path.unlink()
        return False
