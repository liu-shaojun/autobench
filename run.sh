#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Dependency check / install
python3 -c "import yaml, rich, requests" 2>/dev/null || {
    echo "[autobench] installing missing deps (pyyaml, rich, requests)..."
    pip install --quiet pyyaml rich requests
}

exec python3 -m autobench.main "$@"
