"""Fail-closed process audit for the single loaded ACLSHMEM runtime image."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys


_SHMEM_SONAME = re.compile(r"^libshmem\.so(?:\..*)?$")
_MAPS_ESCAPE = re.compile(r"\\([0-7]{3})")


def _decode_maps_path(value: str) -> str:
    """Decode Linux procfs octal escapes such as ``\\040`` in pathnames."""
    return _MAPS_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _is_overlay_device_alias(path: Path, stat_device: int) -> bool:
    """Return whether ``path`` is on an overlay mount represented by stat_device.

    Linux may report the backing filesystem device in ``/proc/self/maps`` while
    ``stat(2)`` reports the container's overlay virtual device.  The inode is
    still stable across both views.  This helper only recognizes that specific
    mount topology; it does not relax inode validation.
    """
    try:
        resolved = path.resolve(strict=True)
        lines = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8", errors="surrogateescape"
        ).splitlines()
    except OSError:
        return False

    device = f"{os.major(stat_device)}:{os.minor(stat_device)}"
    best_match = (-1, False)

    for line in lines:
        left, separator, right = line.partition(" - ")
        fields = left.split()
        fs_fields = right.split()
        if not separator or len(fields) < 5 or not fs_fields:
            continue
        if fields[2] != device:
            continue

        mount_point = Path(_decode_maps_path(fields[4]))
        if resolved.is_relative_to(mount_point):
            length = len(str(mount_point))
            if length > best_match[0]:
                best_match = (length, fs_fields[0] == "overlay")

    return best_match[1]


def _sha256_file(path: Path, expected_identity: tuple[int, int]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != expected_identity:
            raise RuntimeError(
                f"libshmem identity changed before hashing: {path}")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if ((after.st_dev, after.st_ino) != expected_identity or
                after.st_size != before.st_size or
                after.st_mtime_ns != before.st_mtime_ns):
            raise RuntimeError(
                f"libshmem changed while hashing: {path}")
    return digest.hexdigest()


def audit_loaded_shmem_library(
    maps_path: str | os.PathLike[str] = "/proc/self/maps",
    *,
    platform: str | None = None,
) -> dict[str, object]:
    """Return identity evidence for the one loaded ``libshmem.so`` image.

    Mappings are deduplicated by the resolved file's ``(st_dev, st_ino)``;
    repeated ELF segments are therefore one library.  The procfs device/inode
    fields must also match ``stat(2)``.  Any missing, deleted, inconsistent, or
    multiple runtime images fail closed before ACLSHMEM initialization.
    """
    current_platform = sys.platform if platform is None else platform
    if current_platform != "linux":
        raise RuntimeError(
            "libshmem runtime identity audit is unsupported on platform "
            f"{current_platform!r}; Linux /proc/self/maps is required")

    maps_file = Path(maps_path)
    try:
        lines = maps_file.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"cannot audit loaded libshmem runtime: {maps_file} is unavailable"
        ) from exc

    identities: dict[tuple[int, int], dict[str, object]] = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        _, _, _, maps_device, maps_inode_text, raw_path = fields
        mapped_path = _decode_maps_path(raw_path)
        deleted = mapped_path.endswith(" (deleted)")
        candidate_path = mapped_path[:-10] if deleted else mapped_path
        if not _SHMEM_SONAME.match(Path(candidate_path).name):
            continue
        if deleted:
            raise RuntimeError(
                f"loaded libshmem mapping has been deleted: {mapped_path}")
        if not candidate_path.startswith("/"):
            raise RuntimeError(
                f"loaded libshmem mapping has no auditable absolute path: {mapped_path}")

        path = Path(candidate_path)
        try:
            realpath = path.resolve(strict=True)
            stat_result = realpath.stat()
            maps_major_text, maps_minor_text = maps_device.split(":", 1)
            maps_device_number = os.makedev(
                int(maps_major_text, 16), int(maps_minor_text, 16))
            maps_inode = int(maps_inode_text)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot resolve loaded libshmem mapping identity: {mapped_path}") from exc

        inode_matches = maps_inode == stat_result.st_ino
        device_matches = maps_device_number == stat_result.st_dev
        overlay_alias = (
            inode_matches and not device_matches and
            _is_overlay_device_alias(realpath, stat_result.st_dev)
        )
        if not inode_matches or (not device_matches and not overlay_alias):
            raise RuntimeError(
                "loaded libshmem mapping metadata changed during audit: "
                f"{mapped_path} maps dev/inode={maps_device}/{maps_inode}, "
                f"stat dev/inode={os.major(stat_result.st_dev):x}:"
                f"{os.minor(stat_result.st_dev):x}/{stat_result.st_ino}")

        identity = (stat_result.st_dev, stat_result.st_ino)
        identities.setdefault(identity, {
            "realpath": str(realpath),
            "device": int(stat_result.st_dev),
            "device_major": int(os.major(stat_result.st_dev)),
            "device_minor": int(os.minor(stat_result.st_dev)),
            "inode": int(stat_result.st_ino),
            "overlay_device_alias": bool(overlay_alias),
        })

    if len(identities) != 1:
        paths = sorted(str(info["realpath"]) for info in identities.values())
        raise RuntimeError(
            "expected exactly one loaded libshmem.so runtime image, found "
            f"{len(identities)}: {paths}")

    info = next(iter(identities.values()))
    identity = (int(info["device"]), int(info["inode"]))
    try:
        info["sha256"] = _sha256_file(Path(str(info["realpath"])), identity)
    except OSError as exc:
        raise RuntimeError(
            f"cannot hash loaded libshmem runtime: {info['realpath']}") from exc
    return info
