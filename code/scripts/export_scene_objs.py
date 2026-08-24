#!/usr/bin/env python
"""Export ReRoom scenes as posed .obj directories for another paper's renderer.

InstructScene evaluates with FID / KID / SCA computed over Blender renders, and
ships the *real* 3D-FRONT renders it measures against.  That makes those three
columns of its Table 1 genuinely comparable -- but only if this project's
scenes go through the same renderer, at the same resolution, from the same
views.  Their `blender_script.py` consumes a directory of `.obj` files, so this
writes one such directory per scene, with every object already at its final
pose in world coordinates.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.core.scene import Scene


def _asset_dir(jid: str, roots: list[str]) -> str | None:
    for r in roots:
        p = os.path.join(r, jid)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "raw_model.obj")):
            return p
    return None


def export_scene(scene: Scene, out_dir: str, roots: list[str],
                 floor_textures: str | None = None, seed: int = 0) -> int:
    """One posed, textured .obj per kept object, plus the floor.

    This follows InstructScene's own `get_textured_objects` / `export_scene`
    step for step, because a hand-rolled version did not survive their Blender
    script: it rendered as unlit black silhouettes.  Three details turn out to
    matter and none of them are obvious.

    * 3D-FUTURE's ``.mtl`` carries a near-black diffuse constant and trimesh
      does not attach the texture image when it parses it, so the image is set
      explicitly on the material -- exactly what their line
      ``tr_mesh.visual.material.image = Image.open(...)`` does.
    * The scene stays **y-up**, 3D-FRONT's native convention, which is what
      their renderer's camera orbit and lighting assume.  Converting to z-up
      first is what produced the silhouettes.
    * The floor is part of the scene.  Their renders have one; a room without
      it is a floating pile of furniture and no FID computed against them
      means anything.
    """
    import trimesh
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    meshes, names = [], []
    for k, o in enumerate(scene.objects):
        if not o.keep:
            continue
        d = _asset_dir(o.jid, roots) if o.jid else None
        if d is None:
            continue
        tex = os.path.join(d, "texture.png")
        try:
            m = trimesh.load(os.path.join(d, "raw_model.obj"), force="mesh")
            if os.path.exists(tex) and hasattr(m.visual, "material"):
                m.visual.material.image = Image.open(tex)
        except Exception:
            continue
        v = np.asarray(m.vertices, dtype=np.float64)
        if len(v) == 0:
            continue
        # the asset is y-up; ReRoom stores (width, depth, height) as (x, y, z)
        ext = np.maximum(v.max(0) - v.min(0), 1e-6)
        v = v * (np.array([o.size[0], o.size[2], o.size[1]]) / ext)
        v -= (v.max(0) + v.min(0)) / 2.0
        c, sn = math.cos(o.yaw), math.sin(o.yaw)
        R = np.array([[c, 0.0, -sn], [0.0, 1.0, 0.0], [sn, 0.0, c]])
        v = v.dot(R)
        v += np.array([o.xy[0], o.z + o.size[2] / 2.0, o.xy[1]])
        m.vertices = v
        meshes.append(m)
        names.append(f"object_{k:02d}.obj")

    if not meshes:
        return 0

    # ---- floor, y-up, at y = 0 ----
    poly = scene.room.polygon[:, :2]
    from shapely.geometry import Polygon as _P
    from shapely.ops import triangulate
    tri = [t for t in triangulate(_P(poly)) if _P(poly).contains(t.centroid)]
    fv, ff = [], []
    for t in tri:
        cs = np.asarray(t.exterior.coords)[:3]
        base = len(fv)
        for x, z in cs:
            fv.append([x, 0.0, z])
        ff.append([base, base + 1, base + 2])
    if fv:
        fv = np.asarray(fv, dtype=np.float64)
        floor = trimesh.Trimesh(fv, np.asarray(ff), process=False)
        uv = (fv[:, [0, 2]] - fv[:, [0, 2]].min(0))
        uv = uv / max(uv.max(), 1e-6)
        texs = sorted(glob.glob(os.path.join(floor_textures, "*.jpg"))) \
            if floor_textures else []
        if texs:
            t = texs[int(rng.integers(0, len(texs)))]
            floor.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                material=trimesh.visual.material.SimpleMaterial(
                    image=Image.open(t)))
        meshes.append(floor)
        names.append("object_floor.obj")

    # ---- write, using their naming so the .mtl/.png references resolve ----
    for i, m in enumerate(meshes):
        try:
            obj_out, tex_out = trimesh.exchange.obj.export_obj(
                m, return_texture=True)
        except Exception:
            continue
        mtl = f"material_{i:02d}"
        with open(os.path.join(out_dir, names[i]), "w") as fh:
            fh.write(obj_out.replace("material.mtl", mtl + ".mtl")
                     .replace("material_0.mtl", mtl + ".mtl"))
        if tex_out is None:
            continue
        mk = next((k for k in tex_out if k.endswith(".mtl")), None)
        if mk is None:
            continue
        tk = next((k for k in tex_out if not k.endswith(".mtl")), None)
        ext = os.path.splitext(tk)[1] if tk else ".png"
        if tk:
            with open(os.path.join(out_dir, mtl + ext), "wb") as fh:
                fh.write(tex_out[tk])
        # trimesh names the exported texture after the *material*, not
        # `material_0`, so the upstream string swap silently misses whenever
        # the asset's .mtl already had a name -- and 3D-FUTURE's all do
        # (`solid_001_wire`).  Blender then renders the classic missing-texture
        # magenta.  Rewriting the map line directly is immune to the name.
        body = tex_out[mk].decode("utf-8", errors="ignore")
        out_lines = []
        for ln in body.splitlines():
            q = ln.split()
            if q and q[0] in ("map_Kd", "map_Ka"):
                out_lines.append(f"{q[0]} {mtl}{ext}")
            else:
                out_lines.append(ln)
        with open(os.path.join(out_dir, mtl + ".mtl"), "w") as fh:
            fh.write("\n".join(out_lines) + "\n")
    return len(meshes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="directory of scene JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--future-root", required=True, nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--floor-textures", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.scenes, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if a.limit:
        files = files[:a.limit]
    ok = 0
    for k, f in enumerate(files):
        try:
            sc = Scene.load(f)
        except Exception:
            continue
        d = os.path.join(a.out, os.path.splitext(os.path.basename(f))[0])
        n = export_scene(sc, d, a.future_root, a.floor_textures, seed=k)
        if n:
            ok += 1
        if k % 25 == 0:
            print(f"  {k}/{len(files)}  exported={ok}", flush=True)
    print(f"{ok}/{len(files)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
