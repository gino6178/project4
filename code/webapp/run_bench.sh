#!/usr/bin/env bash
# Keep the bench up: restart it if it ever exits, so a crash does not end the
# session silently.  Logs to outputs/bench.log, PID in outputs/bench.pid.
set -u
cd "$(dirname "$0")/.."
export PYOPENGL_PLATFORM=egl PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/gino/miniconda3/envs/reroom/bin/python}
while true; do
  "$PY" webapp/server.py "$@" >> outputs/bench.log 2>&1
  echo "[$(date +%T)] server exited ($?), restarting in 3s" >> outputs/bench.log
  sleep 3
done
