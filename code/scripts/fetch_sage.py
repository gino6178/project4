#!/usr/bin/env python
"""Download a slice of SAGE-10k scene archives (plan section 3.2).

Only the layout JSONs are needed for the augmentation role the plan assigns to
SAGE, so this fetches whole scene zips but keeps just their layouts unless
``--keep-assets`` is passed -- ~50-90 MB per scene otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile

REPO = "nvidia/SAGE-10k"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/scenes?limit=1000"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
UA = {"User-Agent": "Mozilla/5.0 (reroom dataset fetcher)"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--keep-assets", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layouts = os.path.join(a.out, "layouts")
    os.makedirs(layouts, exist_ok=True)

    entries = json.loads(_get(API).decode())
    files = [e["path"] for e in entries if e["path"].endswith(".zip")]
    # a scene's layout is named after the archive, so what is already on disk
    # can be skipped without re-downloading 50 MB to find out
    have = {os.path.basename(p)[:-5] for p in os.listdir(layouts)
            if p.endswith(".json")}
    files = [p for p in files
             if os.path.basename(p)[:-4].split("_", 2)[-1] not in have][:a.n]
    print(f"{len(files)} scene archives to fetch ({len(have)} already local)")
    for k, path in enumerate(files):
        dst = os.path.join(a.out, os.path.basename(path))
        if not os.path.exists(dst):
            with open(dst, "wb") as fh:
                fh.write(_get(RAW + path))
        try:
            with zipfile.ZipFile(dst) as zf:
                for name in zf.namelist():
                    if name.endswith(".json") and "layout_" in name:
                        with zf.open(name) as src, \
                             open(os.path.join(layouts,
                                               os.path.basename(name)), "wb") as out:
                            shutil.copyfileobj(src, out)
        except zipfile.BadZipFile:
            print(f"  bad zip: {path}")
        if not a.keep_assets:
            os.remove(dst)
        if k % 10 == 0:
            print(f"  {k}/{len(files)}", flush=True)
    print("layouts ->", layouts)


if __name__ == "__main__":
    main()
