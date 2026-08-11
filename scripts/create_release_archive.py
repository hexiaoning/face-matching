from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_archive(source: Path, archive: Path) -> Path:
    source = source.resolve()
    archive = archive.resolve()
    if not source.is_dir() or source.parent != archive.parent:
        raise ValueError("source directory and archive must be siblings in the release directory")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    manifest = source / "SHA256SUMS.txt"
    lines = [f"{sha256(path)}  {path.relative_to(source).as_posix()}" for path in files if path != manifest]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as output:
        for path in files:
            output.write(path, Path(source.name) / path.relative_to(source))
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Zip64 offline release with checksums")
    parser.add_argument("source", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = create_archive(args.source, args.archive)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
