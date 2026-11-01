"""Read-time relocation of recorded paths; frozen evidence is never rewritten."""
from __future__ import annotations
from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOTS = ("D:/v6-mission-morph", "D:/v5-final", "D:/final-DQN-v5")
_PATH_FIELDS = frozenset(("path", "source_path", "fixed_source", "mat_path", "native_mat", "native_geometry_mat", "native_geometry_path", "geometry_mat", "mat_output", "audit_path", "case_dir", "evidence", "mesh", "run_dirs", "configs", "stl"))

def resolve_project_path(value, root=PROJECT_ROOT):
    """Map known old roots and project-relative paths to an absolute runtime path.

    Unknown absolute paths (MATLAB/SU2/Gmsh installations) are left external.
    """
    root = Path(root).resolve()
    text = str(value).replace("\\", "/")
    low = text.lower().rstrip("/")
    for prefix in LEGACY_ROOTS:
        old = prefix.lower()
        if low == old or low.startswith(old + "/"):
            text = text[len(prefix):].lstrip("/")
            break
    else:
        if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
            return Path(text)
    if text == "teacher_matlab" or text.startswith("teacher_matlab/"):
        text = "matlab" + text[len("teacher_matlab"):]
    return (root / text).resolve()

def runtime_view(value, root=PROJECT_ROOT, field=None):
    """Return a relocated copy of JSON/CSV data, retaining all non-path values.

    A field named path may be a morph-route label: relative identifiers without
    a slash are not paths. Hashes, dictionary keys and original input bytes stay
    unchanged. Explicit resolve_project_path handles standalone relative files.
    """
    if isinstance(value, dict):
        return {k: runtime_view(v, root, k) for k, v in value.items()}
    if isinstance(value, list):
        return [runtime_view(v, root, field) for v in value]
    if isinstance(value, tuple):
        return tuple(runtime_view(v, root, field) for v in value)
    if isinstance(value, str) and value:
        text = value.replace("\\", "/")
        low = text.lower()
        if any(low == p.lower() or low.startswith(p.lower() + "/") for p in LEGACY_ROOTS):
            return str(resolve_project_path(value, root))
        is_field = field in _PATH_FIELDS or (isinstance(field, str) and (field.endswith("_path") or field.endswith("_mat")))
        if is_field and ("/" in text or Path(text).suffix in (".mat", ".json", ".csv", ".m", ".yaml", ".py")):
            return str(resolve_project_path(value, root))
    return value
