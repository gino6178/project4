#!/usr/bin/env python
"""Write GenRecon `Sage_gt` inputs for ReRoom reference rooms (plan section 3.3).

GenRecon reconstructs a scene from *posed multi-view* images, which is the case
the plan reserves for "when the reference has several photographs".  Its SAGE
mode expects two things, both of which ReRoom can produce from a parsed room:

    renders_room/<id>/transforms.json   Blender/NeRF cameras + image paths
    rooms_raw/<id>/layout_<id>.json     the room's width and length in metres

The room is translated so its bounding box starts at the origin, because the
SAGE chunker assumes a floor plan spanning ``(0, 0)`` to ``(width, length)``.
The same translation is recorded so the reconstruction can be brought back into
the reference room's own frame afterwards.

Instance masks are written alongside each view.  GenRecon returns scene
geometry, not object instances, so lifting instances back out of it needs a
segmentation signal; supplying exact masks isolates the reconstruction quality
from the segmentation quality, exactly as in the MIDI experiment.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np

from reroom.core.scene import Room, Scene
from reroom.render.textured import (camera_poses, load_room_assets, render_room)


def ring_cameras(room: Room, n: int, eye_height: float = 1.55,
                 radius_frac: float = 0.34) -> list[np.ndarray]:
    """Cameras on a ring inside the room, each looking across it.

    A single corner view is enough for a single-image parser; a multi-view
    reconstructor needs coverage, so the views are spread around the room's
    interior rather than picked for how much furniture each one shows.
    """
    from reroom.geom.polygon import as_polygon
    from reroom.render.textured import _look_at
    poly = as_polygon(room)
    c = np.asarray(poly.centroid.coords[0])
    ext = room.extent
    r = radius_frac * float(min(ext))
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        eye = c + np.array([math.cos(a), math.sin(a)]) * r
        tgt = c - np.array([math.cos(a), math.sin(a)]) * r * 0.9
        out.append(_look_at(np.array([eye[0], eye[1], eye_height]),
                            np.array([tgt[0], tgt[1], 0.95])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="outputs/references")
    ap.add_argument("--front", required=True)
    ap.add_argument("--future", required=True, nargs="+")
    ap.add_argument("--out", required=True, help="GenRecon dataset root")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--yfov-deg", type=float, default=70.0)
    a = ap.parse_args()

    renders = os.path.join(a.out, "renders_room")
    rooms_raw = os.path.join(a.out, "rooms_raw")
    os.makedirs(renders, exist_ok=True)
    os.makedirs(rooms_raw, exist_ok=True)

    metas = sorted(glob.glob(os.path.join(a.refs, "*", "meta.json")))[:a.n]
    print(f"{len(metas)} reference rooms", flush=True)
    manifest = []
    for mp in metas:
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        key = meta["scene_id"]
        sid = key.split("__")[0][:8] + "_" + key.split("__")[1].replace("-", "")[:10]
        ref = Scene.load(os.path.join(d, "reference_scene.json"))

        # SAGE floor plans start at the origin
        shift = -ref.room.polygon.min(axis=0)
        room = ref.room.copy()
        room.polygon = room.polygon + shift
        ext = room.extent

        assets = load_room_assets(os.path.join(a.front, f"{meta['house']}.json"),
                                  meta["room"], a.future,
                                  only_oids={o.oid for o in ref.objects})
        if assets is None:
            print(f"  skip {key}: no assets")
            continue
        for _, m in assets.objects:
            m.apply_translation([shift[0], shift[1], 0.0])
        for m in assets.shell:
            m.apply_translation([shift[0], shift[1], 0.0])
        assets.room = room

        rd = os.path.join(renders, sid)
        os.makedirs(rd, exist_ok=True)
        frames = []
        yfov = math.radians(a.yfov_deg)
        aspect = a.width / a.height
        xfov = 2 * math.atan(math.tan(yfov / 2) * aspect)
        for vi, cam in enumerate(ring_cameras(room, a.views)):
            res = render_room(assets, cam, a.width, a.height, yfov=yfov)
            imageio.imwrite(os.path.join(rd, f"{vi:04d}.png"), res.rgb)
            lab = np.zeros(res.instance.shape, np.uint8)
            for idx, oid in enumerate(res.ids):
                lab[res.instance == idx] = idx + 1
            imageio.imwrite(os.path.join(rd, f"{vi:04d}_seg.png"), lab)
            frames.append({"file_path": f"{vi:04d}.png",
                           "transform_matrix": cam.tolist(),
                           "camera_angle_x": float(xfov)})
        with open(os.path.join(rd, "transforms.json"), "w") as fh:
            json.dump({"camera_angle_x": float(xfov), "frames": frames}, fh, indent=1)

        lr = os.path.join(rooms_raw, sid)
        os.makedirs(lr, exist_ok=True)
        with open(os.path.join(lr, f"layout_{sid}.json"), "w") as fh:
            json.dump({"id": sid, "rooms": [{
                "id": sid, "room_type": room.room_type,
                "dimensions": {"width": float(ext[0]), "length": float(ext[1]),
                               "height": float(room.height)}}]}, fh, indent=1)

        manifest.append({"sage_id": sid, "scene_id": key,
                         "shift": shift.tolist(), "views": a.views,
                         "instance_oids": res.ids,
                         "width": float(ext[0]), "length": float(ext[1])})
        print(f"  {key} -> {sid}  {a.views} views  "
              f"{ext[0]:.1f}x{ext[1]:.1f} m", flush=True)

    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
