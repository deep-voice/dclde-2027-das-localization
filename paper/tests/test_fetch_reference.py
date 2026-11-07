import hashlib
import zipfile

import pytest

from paper.scripts import fetch_reference_data as frd
from paper.scripts.fetch_reference_data import colon_safe, read_manifest, verify_checksum, write_manifest


def test_colon_rename_matches_the_dataset_timestamp_normalisation():
    # The dataset writes the same timestamp three ways; underscores are the
    # canonical one (src/dclde_localization/physics/picks.py already normalises to it), and NTFS forbids
    # colons outright, so this is both a convention and a platform requirement.
    assert colon_safe("data/detections/peaks_tp_North_2021-11-04_02:00:02_ipi3.nc") == \
        "data/detections/peaks_tp_North_2021-11-04_02_00_02_ipi3.nc"


def test_colon_rename_leaves_clean_paths_alone():
    assert colon_safe("data/north_DAS_multicoord.csv") == "data/north_DAS_multicoord.csv"


def test_checksum_detects_a_corrupt_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    assert verify_checksum(p, good) is True
    assert verify_checksum(p, hashlib.sha256(b"goodbye").hexdigest()) is False


def test_manifest_round_trips(tmp_path):
    rows = [{"upstream_path": "a/b:c.nc", "local_path": "a/b_c.nc",
             "renamed": True, "bytes": 12},
            {"upstream_path": "x.csv", "local_path": "x.csv", "renamed": False, "bytes": 3}]
    p = tmp_path / "m.csv"
    write_manifest(rows, p)
    got = read_manifest(p)
    assert got == rows


def _write_synthetic_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


def test_materialise_zip_renames_colons_and_writes_manifest(tmp_path, monkeypatch):
    # No network involved: build a small zip with the same colon-in-filename
    # shape as the real Zenodo archive (including a nested directory), point
    # materialise_zip at it, and check the rename/extraction/manifest chain
    # end to end. This is the half of the fetch script that has no built-in
    # integrity check of its own (unlike the git clone), so it is the one
    # whose failure mode is silent if the extraction logic is ever wrong.
    entries = {
        "batch1/association_2021-11-04_02:00:02.h5": b"payload-a",
        "batch1/nested/dir_2021-11-04_02:06:42.h5": b"payload-b",
        "readme.txt": b"no colons here",
    }
    archive = tmp_path / "associations.zip"
    _write_synthetic_zip(archive, entries)
    # materialise_zip only downloads when the archive is absent; here it's
    # already present, so this also exercises the "skip the download, verify
    # what's on disk" path with a digest that matches this specific file.
    monkeypatch.setattr(frd, "ASSOCIATIONS_ZIP_SHA256",
                         hashlib.sha256(archive.read_bytes()).hexdigest())

    rows = frd.materialise_zip(tmp_path)

    by_upstream = {r["upstream_path"]: r for r in rows}
    assert by_upstream["batch1/association_2021-11-04_02:00:02.h5"]["local_path"] == \
        "batch1/association_2021-11-04_02_00_02.h5"
    assert by_upstream["batch1/association_2021-11-04_02:00:02.h5"]["renamed"] is True
    assert by_upstream["readme.txt"]["renamed"] is False

    top = tmp_path / "associations" / "batch1" / "association_2021-11-04_02_00_02.h5"
    nested = tmp_path / "associations" / "batch1" / "nested" / "dir_2021-11-04_02_06_42.h5"
    assert top.read_bytes() == b"payload-a"
    assert nested.read_bytes() == b"payload-b"

    manifest = read_manifest(tmp_path / "associations_manifest.csv")
    assert manifest == rows


def test_materialise_zip_refuses_to_extract_on_checksum_mismatch(tmp_path, monkeypatch):
    # A truncated or bit-flipped download still parses as a zip; the checksum
    # is what turns that into a loud failure instead of a silently-accepted
    # partial extraction.
    archive = tmp_path / "associations.zip"
    _write_synthetic_zip(archive, {"x.h5": b"data"})
    monkeypatch.setattr(frd, "ASSOCIATIONS_ZIP_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256"):
        frd.materialise_zip(tmp_path)

    assert not (tmp_path / "associations").exists()
