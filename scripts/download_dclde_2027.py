#!/usr/bin/env python3
"""Download the DCLDE 2027 DAS fin whale localization dataset from the
public noaa-passive-bioacoustic GCS bucket, using the anonymous JSON/media
API (no gcloud/gsutil credentials required).

Resumable: skips files whose local size already matches the remote size.
"""
import json
import os
import sys
import time
import urllib.request

LISTING_PATH = os.path.join(os.path.dirname(__file__), "..", "dclde_2027_listing.json")
PREFIX = "dclde/2027/dclde_2027_das-finwhale-localization/"
DEST_ROOT = os.path.join(os.path.dirname(__file__), "..", "dclde_2027_das-finwhale-localization")

RETRIES = 5
CHUNK = 1024 * 1024


def download_one(item):
    rel_path = item["name"][len(PREFIX):]
    dest = os.path.join(DEST_ROOT, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    remote_size = int(item["size"])

    if os.path.exists(dest) and os.path.getsize(dest) == remote_size:
        return "skip", rel_path, remote_size

    url = item["mediaLink"]
    tmp = dest + ".part"
    for attempt in range(1, RETRIES + 1):
        try:
            downloaded = 0
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
            if downloaded != remote_size:
                raise IOError(f"size mismatch: got {downloaded}, expected {remote_size}")
            os.replace(tmp, dest)
            return "ok", rel_path, remote_size
        except Exception as e:
            print(f"  retry {attempt}/{RETRIES} for {rel_path}: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    return "fail", rel_path, remote_size


def main():
    with open(LISTING_PATH) as f:
        items = json.load(f)

    total = len(items)
    total_bytes = sum(int(i["size"]) for i in items)
    done_bytes = 0
    ok = skip = fail = 0

    print(f"Downloading {total} objects, {total_bytes/1024/1024/1024:.2f} GB total, to {DEST_ROOT}", flush=True)

    for idx, item in enumerate(items, 1):
        status, rel_path, size = download_one(item)
        done_bytes += size
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
        pct = done_bytes / total_bytes * 100
        print(f"[{idx}/{total}] {status:5s} {rel_path} ({size/1024/1024:.1f} MB) -- {pct:.1f}% of total bytes", flush=True)

    print(f"\nDone. ok={ok} skip={skip} fail={fail}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
