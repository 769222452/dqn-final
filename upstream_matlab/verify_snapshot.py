"""Read-only integrity check for the archived MATLAB pipeline sources."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    expected = set()
    total = 0
    lines = []
    for entry in manifest["files"]:
        name = entry["path"]
        path = (root / name).resolve()
        if not path.is_relative_to((root / "source").resolve()):
            errors.append(f"Invalid source path: {name}")
            continue
        if name in expected:
            errors.append(f"Duplicate source path: {name}")
        expected.add(name)
        if not path.is_file():
            errors.append(f"Missing: {name}")
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            errors.append(f"Content mismatch: {name}")
        lines.append(f"{entry['sha256']}  {name}\n")
    actual = {p.relative_to(root).as_posix() for p in (root / "source").rglob("*") if p.is_file()}
    if actual != expected:
        errors.append(f"Source inventory mismatch: {sorted(actual ^ expected)}")
    if len(expected) != manifest["file_count"] or total != manifest["total_bytes"]:
        errors.append("Manifest counts do not match source files")
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != "".join(lines):
        errors.append("SHA256SUMS does not match SOURCE_MANIFEST.json")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(expected)} files, {total} bytes; all SHA-256 hashes match.")


if __name__ == "__main__":
    main()
