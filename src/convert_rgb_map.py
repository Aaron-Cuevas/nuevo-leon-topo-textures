#!/usr/bin/env python3
import argparse, os
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def decode_gray(img):
    # img: uint8 HxWx3/1 → float32 [0,1]
    if img.ndim==3:
        r,g,b = img[...,0], img[...,1], img[...,2]
        m = 0.2126*r + 0.7152*g + 0.0722*b
    else:
        m = img
    return (m/255.0).astype(np.float32)

def decode_terrarium(img):
    # Terrarium RGB → meters
    r = img[...,0].astype(np.float32)
    g = img[...,1].astype(np.float32)
    b = img[...,2].astype(np.float32)
    elev = (r*256.0 + g + b/256.0) - 32768.0
    # Normaliza a [0,1] para mapas de altura "visuales"
    vmin, vmax = np.nanmin(elev), np.nanmax(elev)
    return ((elev - vmin)/max(1e-6, (vmax-vmin))).astype(np.float32)

def sobel_normals(m, strength=1.0, pixel_size=1.0):
    # m: [0,1] float32
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/(8.0*pixel_size)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)/(8.0*pixel_size)
    dzdx = convolve2d(m, kx, mode="same", boundary="symm")*strength
    dzdy = convolve2d(m, ky, mode="same", boundary="symm")*strength
    nx,ny,nz = -dzdx, -dzdy, np.ones_like(m, np.float32)
    norm = np.sqrt(nx*nx+ny*ny+nz*nz)+1e-8
    nx/=norm; ny/=norm; nz/=norm
    rgb = np.stack([nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5], -1)
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ruta a imagen trazada (png/jpg)")
    ap.add_argument("--mode", choices=["gray","terrarium"], default="gray",
                    help="decodificador: gray=luminancia 0..1, terrarium=RGB→metros→0..1")
    ap.add_argument("--strength", type=float, default=1.0, help="intensidad de derivada para normal map")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if args.mode=="gray":
        m = decode_gray(arr)
    else:
        m = decode_terrarium(arr)

    # Export scalar
    scalar = (np.clip(m,0,1)*255).astype(np.uint8)
    Image.fromarray(scalar, "L").save(os.path.join(args.outdir,"scalar.png"))

    # Normal map (nota: puede verse “opaco”; TODO afinado con curvatura/blur)
    nmap = sobel_normals(m, strength=args.strength, pixel_size=1.0)
    Image.fromarray(nmap, "RGB").save(os.path.join(args.outdir,"normal_from_rgb.png"))

    print("Listo:", os.path.join(args.outdir,"scalar.png"), os.path.join(args.outdir,"normal_from_rgb.png"))
if __name__ == "__main__":
    main()
