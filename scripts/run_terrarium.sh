#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python src/generate_textures.py --zoom 11 --azimuth 315 --altitude 45 --strength 1.0 --outdir out
mkdir -p examples
cp -f out/*.png examples/
echo "OK → out/{hillshade.png,normal.png} y copias en examples/"
