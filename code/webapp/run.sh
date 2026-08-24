#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYOPENGL_PLATFORM=egl PYTHONPATH=.
exec "${PYTHON:-python}" webapp/server.py "$@"
