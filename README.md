
# Nuevo León Topo Texture Generator

Genera dos texturas basadas en la topografía real de **Nuevo León**:
- `nuevo_leon_hillshade.png` — textura **blanco y negro** (relieve sombreado) con alpha fuera del estado.
- `nuevo_leon_normal.png` — **normal map RGB** en espacio tangente para motores 3D.
- `nuevo_leon_height_bump.png` — mapa de altura normalizado (gris) para usar con *Bump/Displacement*.

## Requisitos

```bash
python -m venv .venv && source .venv/bin/activate  # o equivalente en tu OS
pip install -r requirements.txt
```

## Uso

```bash
python make_texture.py --zoom 11 --azimuth 315 --altitude 45 --strength 1.0 --outdir out
```

- `--zoom` 10–12 recomendado (más zoom = más resolución, más descarga).
- `--azimuth` dirección del “sol” en grados (315 = noroeste clásico).
- `--altitude` altura del sol (45° equilibrio entre contraste y detalle).
- `--strength` amplifica pendientes en el normal map.

Los datos se descargan automáticamente de:
- **Terrarium elevation tiles** (AWS): `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`
- **GADM 4.1** para el polígono de *Nuevo León* (México) nivel 1.

> Atribuciones: Terrain Tiles © contributors; GADM data © GADM. Úsalo según sus licencias.

## Blender (nodos)

- **Hillshade**: `Image Texture (sRGB)` → `Color` → `Base Color` o `Multiply` encima de tu albedo.
- **Bump/Height**: `Image Texture (Non-Color)` → `Bump` → `Normal` del Principled BSDF.
- **Normal Map**: `Image Texture (Non-Color)` → `Normal Map` → `Normal` del Principled BSDF.
  - Asegúrate de que el `Normal Map` está en **Tangent Space** y la imagen en Non-Color.

La alpha recorta exactamente el estado.

## Nota técnica

- Conversión **Terrarium → elevación (m)**: `elev = (R*256 + G + B/256) - 32768`.
- **Hillshade** via pendiente y aspecto (iluminación lambertiana).
- **Normal map** vía gradiente \(\nabla z\) y normalización \(( -\partial_x z, -\partial_y z, 1)\).


