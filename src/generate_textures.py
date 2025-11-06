#!/usr/bin/env python3
import argparse, io, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
import requests
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt

UA = {"User-Agent": "NL-Topo-Generator/1.3 (terminal repo)"}
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
GADM_MEX_L1 = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MEX_1.json"

def lonlat_to_tile(lon, lat, z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int((lon + 180.0) / 360.0 * (2 ** z))
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (2 ** z))
    return x, y

def tile_bounds(x, y, z):
    n = 2 ** z
    lon1 = x / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lon2 = (x + 1) / n * 360.0 - 180.0
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (lon1, lat2, lon2, lat1)

def fetch_tile(z, x, y):
    r = requests.get(TERRARIUM.format(z=z, x=x, y=y), headers=UA, timeout=60)
    r.raise_for_status()
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)

def stitch_tiles(tiles, xs, ys):
    H = len(ys)*256; W = len(xs)*256
    out = np.zeros((H,W,3), dtype=np.uint8)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            out[j*256:(j+1)*256, i*256:(i+1)*256] = tiles[(x,y)]
    return out

def lonlat_grid(xs, ys, z):
    W = len(xs)*256; H = len(ys)*256
    lon_vec = np.zeros((W,), dtype=np.float64)
    lat_vec = np.zeros((H,), dtype=np.float64)
    for i, x in enumerate(xs):
        a,b,c,d = tile_bounds(x, ys[0], z)
        lon_vec[i*256:(i+1)*256] = np.linspace(a, c, 256, endpoint=False) + (c-a)/512.0
    for j, y in enumerate(ys):
        a,b,c,d = tile_bounds(xs[0], y, z)
        lat_vec[j*256:(j+1)*256] = np.linspace(d, b, 256, endpoint=False) + (b-d)/512.0
    lon = np.tile(lon_vec[None,:], (H,1))
    lat = np.tile(lat_vec[:,None], (1,W))
    return lon, lat

def terrarium_to_elev(rgb):
    r = rgb[...,0].astype(np.float32)
    g = rgb[...,1].astype(np.float32)
    b = rgb[...,2].astype(np.float32)
    return (r*256.0 + g + b/256.0) - 32768.0

def rasterize_mask(lon, lat, polygon):
    H,W = lon.shape
    lon_min,lon_max = lon[0,0], lon[0,-1]
    lat_min,lat_max = lat[-1,0], lat[0,0]
    def ll_to_px(lonv, latv):
        x = (lonv - lon_min)/(lon_max-lon_min)*(W-1)
        y = (lat_max - latv)/(lat_max-lat_min)*(H-1)
        return x,y
    img = Image.new("L", (W,H), 0)
    d = ImageDraw.Draw(img)
    polys = [polygon] if isinstance(polygon, Polygon) else list(polygon.geoms)
    for poly in polys:
        xs,ys = ll_to_px(*np.array(poly.exterior.coords).T)
        d.polygon(list(map(tuple, np.stack([xs,ys],1))), fill=255)
        for interior in poly.interiors:
            xs,ys = ll_to_px(*np.array(interior.coords).T)
            d.polygon(list(map(tuple, np.stack([xs,ys],1))), fill=0)
    return np.array(img, dtype=np.uint8)

def meters_per_pixel_at_zoom(lat_deg, z):
    equ = 40075016.686/(256*2**z)
    return equ*math.cos(math.radians(lat_deg))

def compute_hillshade(elev, cellsize_m, azimuth_deg=315.0, altitude_deg=45.0):
    z = elev.astype(np.float32)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/(8.0*cellsize_m)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)/(8.0*cellsize_m)
    dzdx = convolve2d(z,kx,mode="same",boundary="symm")
    dzdy = convolve2d(z,ky,mode="same",boundary="symm")
    slope = np.arctan(np.hypot(dzdx,dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    az = np.deg2rad(azimuth_deg); alt = np.deg2rad(altitude_deg)
    hs = np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az - aspect)
    return (np.clip(hs,0,1)*255).astype(np.uint8)

def compute_normals(elev, cellsize_m, strength=1.0):
    z = elev.astype(np.float32)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/(8.0*cellsize_m)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)/(8.0*cellsize_m)
    dzdx = convolve2d(z,kx,mode="same",boundary="symm")*strength
    dzdy = convolve2d(z,ky,mode="same",boundary="symm")*strength
    nx,ny,nz = -dzdx, -dzdy, np.ones_like(z)
    norm = np.sqrt(nx*nx+ny*ny+nz*nz)+1e-8
    nx/=norm; ny/=norm; nz/=norm
    rgb = np.stack([nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5], -1)
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

def fetch_from_osm():
    url = "https://nominatim.openstreetmap.org/search.php"
    params = {"q":"Nuevo León, Mexico","polygon_geojson":1,"format":"jsonv2"}
    r = requests.get(url, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    polys=[]
    for it in r.json():
        gj = it.get("geojson")
        if not gj: continue
        cls = it.get("class"); typ = it.get("type","")
        name = it.get("display_name","")
        if cls=="boundary" and "administrative" in typ and ("Nuevo León" in name or "Nuevo Leon" in name):
            polys.append(shape(gj))
    if not polys:
        for it in r.json():
            gj = it.get("geojson")
            if gj and gj.get("type") in ("Polygon","MultiPolygon"):
                polys.append(shape(gj)); break
    if not polys:
        raise SystemExit("No se pudo obtener el polígono desde OSM/Nominatim.")
    return unary_union(polys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=11)
    ap.add_argument("--azimuth", type=float, default=315.0)
    ap.add_argument("--altitude", type=float, default=45.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[1/6] Obteniendo polígono Nuevo León…")
    poly = fetch_from_osm()
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    pad=0.05; lon_min-=pad; lat_min-=pad; lon_max+=pad; lat_max+=pad

    print("[2/6] Calculando tiles…")
    x0,y1 = lonlat_to_tile(lon_min, lat_min, args.zoom)
    x1,y0 = lonlat_to_tile(lon_max, lat_max, args.zoom)
    xs = list(range(min(x0,x1), max(x0,x1)+1))
    ys = list(range(min(y0,y1), max(y0,y1)+1))
    print(f"   z={args.zoom} cols={len(xs)} rows={len(ys)}")

    print("[3/6] Descargando tiles…")
    tiles={}
    for y in ys:
        for x in xs:
            tiles[(x,y)] = fetch_tile(args.zoom,x,y)

    print("[4/6] Mosaico y elevación…")
    rgb = stitch_tiles(tiles,xs,ys)
    elev = terrarium_to_elev(rgb)
    lon, lat = lonlat_grid(xs,ys,args.zoom)

    print("[5/6] Máscara estatal…")
    mask = rasterize_mask(lon, lat, poly)
    elev_masked = np.where(mask>0, elev, np.nan)
    nan_mask = np.isnan(elev_masked)
    elev_filled = elev_masked.copy()
    if np.any(nan_mask):
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        elev_filled[nan_mask] = elev_masked[tuple(idx[:, nan_mask])]

    print("[6/6] Hillshade y normal…")
    mpp = meters_per_pixel_at_zoom(float(np.nanmean(lat)), args.zoom)
    hs = compute_hillshade(elev_filled, mpp, args.azimuth, args.altitude)
    nm = compute_normals(elev_filled, mpp, args.strength)
    alpha = (mask>0).astype(np.uint8)*255
    hs_rgba = np.dstack([hs,hs,hs,alpha]).astype(np.uint8)
    nm_rgba = np.dstack([nm, alpha[...,None]]).astype(np.uint8)

    Image.fromarray(hs_rgba,"RGBA").save(os.path.join(args.outdir,"hillshade.png"))
    Image.fromarray(nm_rgba,"RGBA").save(os.path.join(args.outdir,"normal.png"))
    print("Listo: out/hillshade.png, out/normal.png")

if __name__ == "__main__":
    main()
