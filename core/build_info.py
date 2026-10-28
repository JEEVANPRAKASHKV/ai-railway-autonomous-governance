"""
core/build_info.py
==================
A fingerprint of the code and data the API serves, so the Streamlit launcher can tell when the running backend
is out of date (Streamlit Cloud pulls new code into the running process on every push, but an API that is already
running keeps the old modules, models and timetable in memory until it is restarted).

The fingerprint is the checked-out git commit plus the source and reference-data files. Model artifacts and the
registry are left out on purpose: the governance layer retrains and promotes models at run time, and that must not
look like a new deployment.
"""

from __future__ import annotations

import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WATCHED = {
    "backend": (".py",), "core": (".py",), "modules": (".py",), "governance": (".py",),
    os.path.join("data", "raw"): (".csv", ".txt"), os.path.join("data", "real"): (".csv", ".gz"),
    os.path.join("data", "knowledge"): (".json",),
}


def git_commit(root: str = ROOT) -> str:
    """The checked-out commit, read straight from .git (no git binary needed); '' if unknown."""
    try:
        git = os.path.join(root, ".git")
        head = open(os.path.join(git, "HEAD"), encoding="utf-8").read().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1].strip()
        p = os.path.join(git, ref)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()[:12]
        with open(os.path.join(git, "packed-refs"), encoding="utf-8") as f:
            for line in f:
                if line.strip().endswith(ref):
                    return line.split()[0][:12]
    except Exception:
        pass
    return ""


def fingerprint(root: str = ROOT) -> str:
    """Short hash of the git commit and every served source / reference-data file (path, size, mtime)."""
    h = hashlib.sha1(git_commit(root).encode())
    for sub, suffixes in _WATCHED.items():
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if not name.endswith(suffixes):
                    continue
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, root)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:12]
