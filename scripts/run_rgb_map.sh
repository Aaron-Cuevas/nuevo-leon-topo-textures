#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
if [ $# -lt 1 ]; then
  echo "Uso: $0 ruta/a/imagen.png [gray|terrarium] [strength]"
  exit 1
fi
IMG="$1"
MODE="${2:-gray}"
STRENGTH="${3:-1.0}"
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python src/convert_rgb_map.py --input "$IMG" --mode "$MODE" --strength "$STRENGTH" --outdir out
mkdir -p examples
cp -f out/*scalar*.png examples/ 2>/dev/null || true
cp -f out/*normal*.png examples/ 2>/dev/null || true
echo "OK → out/{scalar.png,normal_from_rgb.png} y copias en examples/"
