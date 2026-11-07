# Reference data provenance

Not committed (`data/` is gitignored) and deleted from local machines once the
experiments consuming it have produced committed result tables. Restore both
artifacts with:

    uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference

Restore only the reference implementation, skipping the 3.4 GB association
archive:

    uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference --skip-zenodo

| Artifact | Source | Size | Licence |
|---|---|---|---|
| Reference implementation | `github.com/Ocean-Data-Lab/Goestchel_JASA_2025b` | 204 MB | GPL-3.0 |
| Association output, 5 batches | Zenodo record `21220346` | 3.67 GB | see record |

Paper: Goestchel, Q., Wilcock, W. S. D. & Abadi, S. (2026). *JASA* 160(1), 135–146.
doi:10.1121/10.0044257

**Integrity guarantees differ between the two artifacts, on purpose.**

- **Reference implementation:** not checksummed separately. Git blobs are
  content-addressed (SHA-1 per object), so a truncated or corrupted clone
  fails git's own integrity checks before the fetch script ever reads a blob
  out of it. `materialise_repo` additionally checks that the `git clone` exit
  code was 0 and that `git ls-tree -r HEAD` can walk the resulting object
  store before treating an existing `.git` directory as reusable on a re-run
  — so an interrupted clone is retried, not silently accepted.
- **Association archive:** a zip download has no equivalent built-in check,
  so the fetch script records the archive's own SHA-256 and verifies it with
  `verify_checksum` before extracting, raising instead of unpacking on a
  mismatch:

      associations.zip sha256: cb60998134a97a581fdc03f64a5e795dd9b2c521d5cc1c0d600c3afefe686eb5

**Colons in filenames.** 32 of the repo's 45 files and all 180 archive members
carry colons (`..._2021-11-04_02:00:02_...`). NTFS forbids them, so `git clone`
fetches the objects and then aborts the checkout wholesale. The fetch script
renames `:` to `_` and writes a manifest of every mapping.
