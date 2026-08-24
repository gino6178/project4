#!/usr/bin/env bash
# Sequential resumable HF download. usage: fetch_hf.sh <destdir> <repo> <file>...
set -u
DEST="$1"; REPO="$2"; shift 2
mkdir -p "$DEST"
for f in "$@"; do
  url="https://huggingface.co/datasets/$REPO/resolve/main/$f"
  echo "[$(date +%T)] fetching $f"
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
       -o "$DEST/$(basename "$f")" "$url" || echo "FAILED $f"
  echo "[$(date +%T)] done $f -> $(du -h "$DEST/$(basename "$f")" 2>/dev/null | cut -f1)"
done
echo "ALL_DOWNLOADS_COMPLETE"
