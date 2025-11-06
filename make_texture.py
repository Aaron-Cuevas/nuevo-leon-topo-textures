
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuevo León topo texture generator:
- Downloads elevation tiles (Mapzen Terrarium format) for a bbox covering Nuevo León
- Downloads the state polygon (GADM level 1 for Mexico) and clips the raster
- Produces:
    * nuevo_leon_hillshade.png (grayscale)
    * nuevo_leon_normal.png (tangent-space normal map, RGB)
Usage:
    python make_texture.py --zoom 11 --azimuth 315 --altitude 45 --outdir out
"""
import argparse, io, math, os, sys, json, time, concurrent.futures, itertools
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw
import requests
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

# -------------------- Config --------------------
# Loose bbox around Nuevo León to ensure coverage; clipped later to polygon.
# (min_lon, min_lat, max_lon, max_lat)
NL_BBOX = (-101.95, 23.0, -98.0, 27.95)

GADM_MEX_L1 = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MEX_1.json"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
UA = {"User-Agent": "NL-Topo-Generator/1.0"}

# -------------------- Helpers --------------------
def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int,int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int((lon + 180.0) / 360.0 * (2 ** z))
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (2 ** z))
    return x, y

def tile_bounds(x: int, y: int, z: int) -> Tuple[float,float,float,float]:
    n = 2 ** z
    lon1 = x / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lon2 = (x + 1) / n * 360.0 - 180.0
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (lon1, lat2, lon2, lat1)  # min_lon, min_lat, max_lon, max_lat

def terrarium_to_elev(arr: np.ndarray) -> np.ndarray:
    # arr shape: [H,W,3], uint8
    r = arr[...,0].astype(np.float32)
    g = arr[...,1].astype(np.float32)
    b = arr[...,2].astype(np.float32)
    elev = (r * 256.0 + g + b / 256.0) - 32768.0
    return elev

def fetch_json(url: str) -> dict:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    url = TERRARIUM.format(z=z, x=x, y=y)
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed {url}: {r.status_code}")
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.array(im, dtype=np.uint8)

def stitch_tiles(tiles: dict, xs: List[int], ys: List[int]) -> np.ndarray:
    # tiles[(x,y)] = np.uint8[256,256,3]
    H = len(ys)*256
    W = len(xs)*256
    out = np.zeros((H,W,3), dtype=np.uint8)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            out[j*256:(j+1)*256, i*256:(i+1)*256] = tiles[(x,y)]
    return out

def lonlat_grid(xs: List[int], ys: List[int], z: int):
    # Returns arrays of lon,lat for pixel centers
    W = len(xs)*256
    H = len(ys)*256
    # build per-tile coordinate vectors to save time
    lon_vec = np.zeros((W,), dtype=np.float64)
    lat_vec = np.zeros((H,), dtype=np.float64)
    # fill lon
    for i, x in enumerate(xs):
        min_lon, min_lat, max_lon, max_lat = tile_bounds(x, ys[0], z)
        # within a tile, lon increases left->right linearly in WebMercator x
        lon_tile = np.linspace(min_lon, max_lon, 256, endpoint=False) + (max_lon - min_lon)/512.0
        lon_vec[i*256:(i+1)*256] = lon_tile
    # fill lat
    for j, y in enumerate(ys):
        min_lon, min_lat, max_lon, max_lat = tile_bounds(xs[0], y, z)
        lat_tile = np.linspace(max_lat, min_lat, 256, endpoint=False) + (min_lat - max_lat)/512.0
        lat_vec[j*256:(j+1)*256] = lat_tile
    lon = np.tile(lon_vec[None, :], (H,1))
    lat = np.tile(lat_vec[:, None], (1,W))
    return lon, lat

def rasterize_mask(lon, lat, polygon: MultiPolygon) -> np.ndarray:
    # naive rasterization via PIL after mapping lon/lat to pixel coords
    H, W = lon.shape
    # map lon/lat to pixel space [0,W), [0,H)
    # Build a transform: we know lon varies per-column according to lon[0,i]; lat varies per-row.
    # We'll use closest match via linear mapping.
    lon_min = lon[0,0]; lon_max = lon[0,-1]
    lat_min = lat[-1,0]; lat_max = lat[0,0]
    def ll_to_px(lonv, latv):
        x = (lonv - lon_min) / (lon_max - lon_min) * (W-1)
        y = (lat_max - latv) / (lat_max - lat_min) * (H-1)
        return x, y
    img = Image.new("L", (W,H), 0)
    draw = ImageDraw.Draw(img)
    if isinstance(polygon, Polygon):
        polys = [polygon]
    else:
        polys = list(polygon.geoms)
    for poly in polys:
        coords = np.asarray(poly.exterior.coords)
        xs, ys = ll_to_px(coords[:,0], coords[:,1])
        pts = list(map(tuple, np.stack([xs, ys], axis=1)))
        draw.polygon(pts, fill=255)
        for interior in poly.interiors:
            icoords = np.asarray(interior.coords)
            xs, ys = ll_to_px(icoords[:,0], icoords[:,1])
            pts = list(map(tuple, np.stack([xs, ys], axis=1)))
            draw.polygon(pts, fill=0)
    return np.array(img, dtype=np.uint8)

def compute_hillshade(elev: np.ndarray, cellsize_m: float, azimuth_deg=315.0, altitude_deg=45.0) -> np.ndarray:
    # Gradient via Horn operator
    z = elev.astype(np.float32)
    # assume square pixels; approximate meters per pixel via latitude ~ mid-lat of NL ~ 25.5 deg
    # Use simple Sobel-like kernel
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32) / (8.0*cellsize_m)
    kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32) / (8.0*cellsize_m)
    from scipy.signal import convolve2d
    dzdx = convolve2d(z, kernel_x, mode="same", boundary="symm")
    dzdy = convolve2d(z, kernel_y, mode="same", boundary="symm")
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    aspect_rad = np.arctan2(-dzdx, dzdy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    hs = np.sin(alt)*np.cos(slope_rad) + np.cos(alt)*np.sin(slope_rad)*np.cos(az - aspect_rad)
    hs = np.clip(hs, 0, 1)
    return (hs*255.0).astype(np.uint8)

def compute_normals(elev: np.ndarray, cellsize_m: float, strength=1.0) -> np.ndarray:
    z = elev.astype(np.float32)
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32) / (8.0*cellsize_m)
    kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32) / (8.0*cellsize_m)
    from scipy.signal import convolve2d
    dzdx = convolve2d(z, kernel_x, mode="same", boundary="symm") * strength
    dzdy = convolve2d(z, kernel_y, mode="same", boundary="symm") * strength
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    rgb = np.stack([(nx*0.5+0.5), (ny*0.5+0.5), (nz*0.5+0.5)], axis=-1)
    return (np.clip(rgb, 0, 1)*255.0).astype(np.uint8)

def meters_per_pixel_at_zoom(lat_deg: float, z: int) -> float:
    # approximate meters per pixel at given latitude for WebMercator
    # source: standard formula
    earth_circumference = 40075016.686
    m_per_pixel_equator = earth_circumference / (256 * 2**z)
    return m_per_pixel_equator * math.cos(math.radians(lat_deg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=11, help="WebMercator zoom level (10-12 good range)")
    ap.add_argument("--azimuth", type=float, default=315.0, help="Sun azimuth in degrees")
    ap.add_argument("--altitude", type=float, default=45.0, help="Sun altitude in degrees")
    ap.add_argument("--strength", type=float, default=1.0, help="Normal map slope amplification")
    ap.add_argument("--outdir", type=str, default="out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Download and parse GADM Mexico L1, extract Nuevo León polygon
    print("[1/6] Downloading GADM L1 for Mexico...")
    gj = fetch_json(GADM_MEX_L1)
    polys = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        name = props.get("NAME_1") or props.get("NAME_1".lower())
        if name and name.strip().lower() in ("nuevo león","nuevo leon"):
            geom = shape(feat["geometry"])
            if isinstance(geom, (Polygon, MultiPolygon)):
                polys.append(geom)
    if not polys:
        print("Could not find Nuevo León polygon in GADM. Aborting.", file=sys.stderr)
        sys.exit(2)
    nl_poly = unary_union(polys)
    nl_bounds = nl_poly.bounds  # lon_min, lat_min, lon_max, lat_max
    # Expand a bit to be safe
    lon_min = min(NL_BBOX[0], nl_bounds[0]) - 0.05
    lat_min = min(NL_BBOX[1], nl_bounds[1]) - 0.05
    lon_max = max(NL_BBOX[2], nl_bounds[2]) + 0.05
    lat_max = max(NL_BBOX[3], nl_bounds[3]) + 0.05

    # 2) Determine tiles
    x0, y1 = lonlat_to_tile(lon_min, lat_min, args.zoom)
    x1, y0 = lonlat_to_tile(lon_max, lat_max, args.zoom)
    xs = list(range(x0, x1+1))
    ys = list(range(y0, y1+1))
    print(f"[2/6] Tile range at z={args.zoom}: X {xs[0]}..{xs[-1]} ({len(xs)} cols), Y {ys[0]}..{ys[-1]} ({len(ys)} rows)")

    # 3) Download tiles
    print("[3/6] Downloading Terrarium tiles...")
    tiles = {}
    def fetch_xy(xy):
        x,y = xy
        for attempt in range(3):
            try:
                arr = fetch_tile(args.zoom, x, y)
                return (xy, arr)
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.2*(attempt+1))
        raise RuntimeError("unreachable")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for (x,y), arr in ex.map(fetch_xy, list(itertools.product(xs, ys))):
            tiles[(x,y)] = arr
    print(f"  Downloaded {len(tiles)} tiles.")

    # 4) Stitch and convert to elevation
    print("[4/6] Stitching tiles and converting to elevation...")
    rgb = stitch_tiles(tiles, xs, ys)
    elev = terrarium_to_elev(rgb)  # meters
    lon, lat = lonlat_grid(xs, ys, args.zoom)

    # 5) Mask to Nuevo León polygon
    print("[5/6] Masking to Nuevo León polygon...")
    mask = rasterize_mask(lon, lat, nl_poly)
    elev_masked = np.where(mask>0, elev, np.nan)

    # 6) Compute cellsize (meters/pixel) at mid-lat
    mid_lat = float(np.nanmean(lat))
    mpp = meters_per_pixel_at_zoom(mid_lat, args.zoom)

    # Fill NaNs by nearest neighbor for derivatives, but keep alpha for export
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(elev_masked)
    elev_filled = elev_masked.copy()
    if np.any(nan_mask):
        print("  Inpainting voids at borders...")
        # nearest-neighbor fill
        dist, idx = distance_transform_edt(nan_mask, return_indices=True)
        elev_filled[nan_mask] = elev_masked[tuple(idx[:, nan_mask])]

    # Hillshade
    print("[6/6] Computing hillshade and normal map...")
    hs = compute_hillshade(elev_filled, cellsize_m=mpp, azimuth_deg=args.azimuth, altitude_deg=args.altitude)
    # Normal map
    normals = compute_normals(elev_filled, cellsize_m=mpp, strength=args.strength)

    # Apply alpha from polygon mask
    alpha = (mask>0).astype(np.uint8)*255
    hs_rgba = np.dstack([hs, hs, hs, alpha]).astype(np.uint8)
    normal_rgba = np.dstack([normals, alpha[...,None]]).astype(np.uint8)

    # Save
    hs_im = Image.fromarray(hs_rgba, mode="RGBA")
    nm_im = Image.fromarray(normal_rgba, mode="RGBA")
    hs_path = os.path.join(args.outdir, "nuevo_leon_hillshade.png")
    nm_path = os.path.join(args.outdir, "nuevo_leon_normal.png")
    hs_im.save(hs_path, optimize=True)
    nm_im.save(nm_path, optimize=True)

    # Also save a pure height (bump) grayscale for convenience (normalized 0-1 inside mask)
    v = elev_masked.copy()
    v_min = np.nanmin(v); v_max = np.nanmax(v)
    v_norm = (v - v_min) / max(1e-6, (v_max - v_min))
    v_gray = (np.nan_to_num(v_norm, nan=0.0)*255.0).astype(np.uint8)
    bump_rgba = np.dstack([v_gray, v_gray, v_gray, alpha]).astype(np.uint8)
    bump_path = os.path.join(args.outdir, "nuevo_leon_height_bump.png")
    Image.fromarray(bump_rgba, mode="RGBA").save(bump_path, optimize=True)

    meta = {
        "bbox_used": [lon.min().item(), lat.min().item(), lon.max().item(), lat.max().item()],
        "zoom": args.zoom,
        "meters_per_pixel": mpp,
        "azimuth_deg": args.azimuth,
        "altitude_deg": args.altitude,
        "strength": args.strength,
        "outputs": {
            "hillshade_png": os.path.abspath(hs_path),
            "normal_png": os.path.abspath(nm_path),
            "height_bump_png": os.path.abspath(bump_path)
        }
    }
    with open(os.path.join(args.outdir, "nuevo_leon_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")
    print("Outputs:")
    print(" -", hs_path)
    print(" -", nm_path)
    print(" -", bump_path)

if __name__ == "__main__":
    main()
