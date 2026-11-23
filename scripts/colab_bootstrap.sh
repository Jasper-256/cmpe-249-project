#!/bin/bash
# Prepare a Colab VM for synthetic-only training (track B).
# Usage on the VM: bash colab_bootstrap.sh <git-commit>
# Expects /content/synthetic-split-v1.json (uploaded from data/fresh/index/).
# Real data never goes to Colab: real evaluation runs locally.
set -euo pipefail
COMMIT="$1"
cd /content
if [ ! -d repo ]; then
  git clone -q https://github.com/Jasper-256/cmpe-249-project repo
fi
cd repo
git fetch -q origin
git checkout -q "$COMMIT"
command -v uv >/dev/null || pip install -q uv
uv sync -q
mkdir -p data/fresh/index
cp /content/synthetic-split-v1.json data/fresh/index/synthetic-split-v1.json
uv run python scripts/download_fresh.py --only synthetic.zip --only models.zip
uv run python scripts/build_crop_cache.py --targets synthetic --workers 2 \
  --output outputs/data/crop-cache-v2-colab.json
echo "bootstrap done $(date)"
