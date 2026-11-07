"""Re-download the reference data this research compares against.

Two artifacts, both public, ~3.6 GB together. They are deliberately NOT
committed (data/ is gitignored) and are deleted from local machines once the
experiments consuming them have produced committed result tables. This script
is the reproducibility guarantee, and it verifies each artifact the way that
artifact actually needs: the repo clone is content-addressed by git, so a
corrupted clone fails git's own checks; the zip download is not
self-verifying, so it is checked against a recorded SHA-256
(ASSOCIATIONS_ZIP_SHA256 below) before being extracted.

Windows cannot check either out directly: 32 of the repo's 45 files and all 180
archive members carry colons in their names, which NTFS forbids. Git aborts the
checkout wholesale. Everything is materialised through a ':' -> '_' rename with
a manifest recording every mapping.

    uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/Ocean-Data-Lab/Goestchel_JASA_2025b.git"
ZENODO_URL = "https://zenodo.org/records/21220346/files/associations.zip?download=1"
MANIFEST_FIELDS = ["upstream_path", "local_path", "renamed", "bytes"]

# The zip is not self-verifying the way a git clone is (see materialise_repo's
# docstring), so its integrity is checked against this recorded digest before
# extraction. Computed once from the archive fetched from ZENODO_URL above;
# also recorded in paper/data/reference/PROVENANCE.md for anyone auditing without
# reading code.
ASSOCIATIONS_ZIP_SHA256 = "cb60998134a97a581fdc03f64a5e795dd9b2c521d5cc1c0d600c3afefe686eb5"


def colon_safe(path: str) -> str:
    """':' -> '_', matching the underscore normalisation src/dclde_localization/physics/picks.py uses."""
    return path.replace(":", "_")


def verify_checksum(path, expected_sha256: str) -> bool:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def write_manifest(rows, path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [{"upstream_path": r["upstream_path"], "local_path": r["local_path"],
                 "renamed": r["renamed"] == "True", "bytes": int(r["bytes"])}
                for r in csv.DictReader(fh)]


def _repo_ls_tree(repo: Path):
    """('%(objectname)', '%(path)') pairs for every blob at HEAD, or None if
    the object store isn't usable -- e.g. an interrupted or failed clone left
    a `.git` directory behind without a full history. Git's content
    addressing means a corrupted clone fails here (or in the `cat-file` calls
    that follow) rather than being silently accepted; that is this artifact's
    integrity guarantee, in place of a separate checksum."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "HEAD", "--format=%(objectname) %(path)"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return [line.split(" ", 1) for line in out.decode().splitlines()]


def materialise_repo(dest: Path) -> list[dict]:
    """Clone bare, then write each blob under a colon-safe name."""
    repo = dest / "Goestchel_JASA_2025b"
    entries = _repo_ls_tree(repo) if (repo / ".git").exists() else None
    if entries is None:
        result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)])
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed (exit {result.returncode}): {REPO_URL}")
        entries = _repo_ls_tree(repo)
        if entries is None:
            raise RuntimeError(
                f"git clone reported success but {repo} has no usable object store "
                "(git ls-tree -r HEAD failed) -- remove the directory and re-run"
            )

    out, rows = repo / "_worktree", []

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a],
                              check=True, stdout=subprocess.PIPE).stdout

    for sha, path in entries:
        safe = colon_safe(path)
        target = out / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("cat-file", "blob", sha))
        rows.append({"upstream_path": path, "local_path": safe,
                     "renamed": path != safe, "bytes": target.stat().st_size})
    write_manifest(rows, repo / "_worktree_manifest.csv")
    return rows


def materialise_zip(dest: Path) -> list[dict]:
    """Download (if needed), verify against ASSOCIATIONS_ZIP_SHA256, then
    extract. Unlike a git clone, a zip download has no built-in integrity
    check, so a truncated or corrupted archive would otherwise unpack without
    complaint -- the checksum is what makes that failure loud instead of
    silent."""
    archive = dest / "associations.zip"
    if not archive.exists():
        subprocess.run(["curl", "-sL", ZENODO_URL, "-o", str(archive)], check=True)
    if not verify_checksum(archive, ASSOCIATIONS_ZIP_SHA256):
        raise RuntimeError(
            f"{archive} failed SHA-256 verification (expected "
            f"{ASSOCIATIONS_ZIP_SHA256}) -- the download may be truncated or "
            "corrupt; refusing to extract. Delete the file and re-run to "
            "re-download."
        )
    out, rows = dest / "associations", []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            safe = colon_safe(info.filename)
            target = out / safe
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src:
                target.write_bytes(src.read())
            rows.append({"upstream_path": info.filename, "local_path": safe,
                         "renamed": info.filename != safe, "bytes": info.file_size})
    write_manifest(rows, dest / "associations_manifest.csv")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="paper/data/reference", type=Path)
    ap.add_argument("--skip-zenodo", action="store_true",
                    help="skip the 3.4 GB association archive")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    rows = materialise_repo(args.dest)
    print(f"reference repo: {len(rows)} files, "
          f"{sum(r['renamed'] for r in rows)} renamed, "
          f"{sum(r['bytes'] for r in rows) / 1e6:.0f} MB")
    if not args.skip_zenodo:
        rows = materialise_zip(args.dest)
        print(f"associations: {len(rows)} files, "
              f"{sum(r['bytes'] for r in rows) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
