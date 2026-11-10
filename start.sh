#!/usr/bin/env bash
# One-click start for Mac / Linux / Raspberry Pi: creates the virtual environment and installs the
# requirements the first time, then starts ExamGuard (restarting it if it stops).
#   ./start.sh                      same options as run.py, e.g.  ./start.sh --https --host 0.0.0.0
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: creating the virtual environment ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f .venv/.requirements-installed ] || [ requirements.txt -nt .venv/.requirements-installed ]; then
  echo "Installing requirements (needs internet, a few minutes) ..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
  touch .venv/.requirements-installed
fi
exec python start.py "$@"
