#!/usr/bin/env python
"""ReRoom bench — a local page for judging retargeting by eye.

Section 14.4 of the plan says the question is not reconstruction accuracy but
whether a person still recognises the target as carrying the reference room's
design.  That is a judgement, not a metric, so this serves the pieces needed to
make it: a reference room, a target floor you shape yourself, the retargeted
result rendered from the same real assets, the direct-scaling baseline beside
it, and the two questions the plan wants asked separately.

    python webapp/server.py --refs outputs/references --port 8000

Everything runs locally; nothing is uploaded.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import math
import os
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Single-threaded on purpose.  The optimiser runs many tiny tensor ops, so
# threading adds synchronisation without work to share: measured 3.58 s at one
# thread against 4.03 s at eight.  A GPU is worse still (8.7 s) -- the kernels
# are too small to fill it and launch overhead dominates.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)
from flask import Flask, jsonify, request, send_from_directory

from reroom.core.scene import Room, Scene
from reroom.data.asset_bank import AssetBank
from reroom.eval.metrics import evaluate
from reroom.geom.deform import (aspect_deform, corner_cut, slant_wall,
                                uniform_scale, validate_polygon)
from reroom.geom.polygon import normalize_polygon
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.textured import (load_room_assets, render_scene_textured,
                                    repose_assets)
from reroom.render.topdown import draw_scene
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.populate import CooccurrenceModel

app = Flask(__name__, static_folder=None)
CFG: dict = {}
_LOCK = threading.Lock()
_ASSETS: dict = {}
_GRAPHS: dict = {}
# One retarget at a time.  A single request runs a batched optimiser over
# ~20 restarts; without this an open URL is a one-line way to take the machine
# down, and the results would be wrong anyway from thrashing.
_WORK = threading.Semaphore(1)
_HITS: dict = {}
# The plan's output is an *editable* scene (eq. 3): object instances with a
# pose, a size and an asset id, not a baked mesh.  Keeping the last result per
# session is what lets the page actually exercise that -- move a chair, turn
# the bed, drop the plant, re-render, export.
_SESSIONS: dict = {}


def _token_ok() -> bool:
    tok = CFG.get("token")
    if not tok:
        return True
    given = (request.headers.get("X-ReRoom-Token")
             or request.args.get("k")
             or (request.get_json(silent=True) or {}).get("k"))
    return given == tok


@app.before_request
def _guard():
    if request.path.startswith("/api/") or request.path == "/":
        if not _token_ok():
            return jsonify({"error": "bad or missing key"}), 403
    if request.path == "/api/retarget":
        ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"
        now = time.time()
        hits = [t for t in _HITS.get(ip, []) if now - t < 60]
        if len(hits) >= CFG.get("rate", 20):
            return jsonify({"error": "too many requests, wait a minute"}), 429
        hits.append(now)
        _HITS[ip] = hits
    return None


# ---------------------------------------------------------------- helpers --
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=118, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img_array(arr, quality: int = 82) -> str:
    """Encode a render as JPEG.

    These are photograph-like images and they travel over a tunnel as base64,
    where PNG costs about five times the bytes for no visible gain.  Line-art
    floor plans stay PNG, where PNG is the right format.
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr[..., :3]).save(buf, format="JPEG", quality=quality,
                                       optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def topdown_png(scene: Scene, title: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    draw_scene(ax, scene, title=title, labels=True, fontsize=7)
    # painted no-go floor, so a zone that was asked for is visibly respected
    for z in getattr(scene.room, "keepout", ()) or ():
        ax.add_patch(plt.Polygon(np.asarray(z)[:, :2], closed=True,
                                 facecolor="#d94848", edgecolor="#a02020",
                                 alpha=0.16, hatch="//", zorder=0.5))
    # a pinned object gets a ring, so "the solver left this alone" is visible
    for o in scene.objects:
        if o.keep and o.locked:
            ax.plot([o.xy[0]], [o.xy[1]], marker="o", ms=11, mfc="none",
                    mec="#1b6ac9", mew=2.0, zorder=6)
    return _png(fig)


def _jpeg_file(path: str, quality: int = 82) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=quality,
                                         optimize=True)
    return buf.getvalue()


def case_dir(case_id: str) -> str:
    return os.path.join(CFG["refs"], case_id)


def load_case(case_id: str):
    """Reference scene, its design-intent graph, and its real geometry."""
    with _LOCK:
        if case_id in _GRAPHS:
            return _GRAPHS[case_id], _ASSETS.get(case_id)
    d = case_dir(case_id)
    scene = Scene.load(os.path.join(d, "reference_scene.json"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    graph = build_motifs(build_scene_graph(scene))
    assets = None
    if CFG.get("front") and CFG.get("future"):
        try:
            assets = load_room_assets(
                os.path.join(CFG["front"], f"{meta['house']}.json"),
                meta["room"], CFG["future"],
                only_oids={o.oid for o in scene.objects})
            if assets is not None:
                assets.room = scene.room
        except Exception as exc:
            print(f"  [assets] {case_id}: {exc}", flush=True)
    with _LOCK:
        _GRAPHS[case_id] = (scene, graph, meta)
        _ASSETS[case_id] = assets
    return (scene, graph, meta), assets


def build_target(room: Room, spec: dict) -> tuple[Room, str]:
    """Turn the page's controls into a target floor polygon."""
    p = room.polygon
    mode = spec.get("mode", "custom")
    label = mode
    if mode == "original":
        poly = uniform_scale(p, 1.0)
    elif mode == "smaller70":
        poly = uniform_scale(p, float(math.sqrt(0.7)))
        label = "70% of the area"
    elif mode == "narrow":
        poly = aspect_deform(p, 0.72, 1.25)
    elif mode == "wide":
        poly = aspect_deform(p, 1.30, 0.95)
    elif mode == "l_shaped":
        poly = corner_cut(p, 0, 0.42, 0.42, 0.0)
    elif mode == "slanted":
        walls = room.walls()
        L = float(np.linalg.norm(walls[1 % len(walls)][1] - walls[1 % len(walls)][0]))
        poly = slant_wall(p, 1, 0.30 * L, "normal")
    else:
        sx = float(spec.get("scale_x", 1.0))
        sy = float(spec.get("scale_y", 1.0))
        poly = aspect_deform(p, sx, sy)
        cut = float(spec.get("corner_cut", 0.0))
        if cut > 0.02:
            poly = corner_cut(normalize_polygon(poly),
                              int(spec.get("corner", 0)), cut, cut,
                              float(spec.get("oblique", 0.0)))
        sl = float(spec.get("slant", 0.0))
        if abs(sl) > 0.02:
            q = normalize_polygon(poly)
            w = int(spec.get("wall", 1)) % len(q)
            L = float(np.linalg.norm(q[(w + 1) % len(q)] - q[w]))
            poly = slant_wall(q, w, sl * L, "normal")
        label = f"{sx:.2f} x {sy:.2f}"
    poly = normalize_polygon(poly)
    if not validate_polygon(poly):
        poly = normalize_polygon(uniform_scale(p, 1.0))
        label += " (invalid — reverted)"
    return Room(polygon=poly, height=room.height,
                openings=[o.copy() for o in room.openings],
                room_type=room.room_type), label


def metrics_payload(graph, scene: Scene) -> dict:
    m = evaluate(graph, scene)
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in m.items() if not isinstance(v, dict)}


# ----------------------------------------------------------------- routes --
@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)),
                               "index.html")


@app.get("/api/cases")
def api_cases():
    out = []
    for mp in sorted(glob.glob(os.path.join(CFG["refs"], "*", "meta.json"))):
        meta = json.load(open(mp))
        d = os.path.dirname(mp)
        try:
            s = Scene.load(os.path.join(d, "reference_scene.json"))
        except Exception:
            continue
        out.append({
            "id": meta["scene_id"],
            "room_type": s.room.room_type.replace("_", " "),
            "area": round(s.room.area, 1),
            "n_objects": len(s.objects),
            "n_vertices": len(s.room.polygon),
            "has_photo": os.path.exists(os.path.join(d, "rgb.png")),
            "categories": [c for c, _ in
                           Counter(o.category for o in s.objects).most_common(4)],
        })
    return jsonify(out)


@app.get("/api/reference/<case_id>")
def api_reference(case_id):
    (scene, graph, meta), assets = load_case(case_id)
    d = case_dir(case_id)
    photo = None
    p = os.path.join(d, "rgb.png")
    if os.path.exists(p):
        photo = ('data:image/jpeg;base64,' + base64.b64encode(
            _jpeg_file(p)).decode())
    render = None
    if assets is not None:
        try:
            _, res = render_scene_textured(assets, scene.room, 760, 540)
            render = _img_array(res.rgb)
        except Exception as exc:
            print(f"  [render] {case_id}: {exc}", flush=True)
    return jsonify({
        "id": case_id,
        "room_type": scene.room.room_type.replace("_", " "),
        "area": round(scene.room.area, 1),
        "n_objects": len(scene.objects),
        "polygon": scene.room.polygon.tolist(),
        "objects": [{"category": o.category,
                     "size": [round(float(x), 2) for x in o.size]}
                    for o in scene.objects],
        "topdown": topdown_png(scene),
        "photo": photo,
        "render": render,
        "motifs": [{"name": m.name,
                    "members": [scene.objects[i].category for i in m.members]}
                   for m in graph.motifs],
    })


@app.post("/api/outline")
def api_outline():
    """The target floor polygon in 0..1 box coordinates.

    The no-go zone painter needs to show the shape the user is about to solve
    into, not the reference -- otherwise a zone drawn in the corner of an
    L-shaped target lands somewhere else entirely.
    """
    body = request.get_json(force=True)
    case_id = body.get("case")
    if case_id not in {os.path.basename(os.path.dirname(m)) for m in
                       glob.glob(os.path.join(CFG["refs"], "*", "meta.json"))}:
        return jsonify({"error": "unknown case"}), 400
    (scene, _, _), _ = load_case(case_id)
    room, label = build_target(scene.room, body.get("target", {}))
    b = room.bbox
    ext = np.maximum(b[1] - b[0], 1e-6)
    pts = (room.polygon[:, :2] - b[0]) / ext
    return jsonify({"label": label, "area": round(room.area, 1),
                    "polygon": [[round(float(x), 4), round(float(y), 4)]
                                for x, y in pts]})


@app.post("/api/retarget")
def api_retarget():
    body = request.get_json(force=True)
    case_id = body["case"]
    if case_id not in {os.path.basename(os.path.dirname(m)) for m in
                       glob.glob(os.path.join(CFG["refs"], "*", "meta.json"))}:
        return jsonify({"error": "unknown case"}), 400
    if not _WORK.acquire(timeout=90):
        return jsonify({"error": "busy — another retarget is running"}), 429
    try:
        return _do_retarget(body, case_id)
    finally:
        _WORK.release()


def _do_retarget(body, case_id):
    (scene, graph, meta), assets = load_case(case_id)
    room, label = build_target(scene.room, body.get("target", {}))
    t0 = time.time()

    # Quality presets, chosen by measurement rather than by feel: the fast one
    # halves the wall clock (1.78 s vs 3.58 s) for a 0.004 change in the joint
    # score, which is well inside run-to-run noise.
    q = body.get("quality", "fast")
    gs, ps, rs = (200, 90, 20) if q == "good" else (100, 45, 12)
    cfg = RetargetConfig(
        restarts=int(body.get("restarts", rs)), device=CFG.get("device", "cpu"),
        grad_steps=gs, proj_steps=ps,
        seed=int(body.get("seed", 0)),
        allow_removal=bool(body.get("allow_removal", True)),
        allow_addition=bool(body.get("allow_addition", True)),
        allow_substitution=bool(body.get("allow_substitution", True)))
    # C_t (section 1): the page can pin objects and paint no-go floor.  Pins
    # are carried on a *copy* of the reference graph so one request cannot
    # leave a pin behind for the next visitor.
    pins = {str(x) for x in body.get("pinned", [])}
    if pins:
        graph = _graph_with_pins(graph, pins)
    zones = _keepout_polygons(room, body.get("keepout", []))
    if zones:
        room = room.copy()
        room.keepout = zones
    res = retarget(graph, room, elasticity=CFG.get("elasticity"),
                   bank=CFG.get("bank"), cooc=CFG.get("cooc"), cfg=cfg)
    ours = res.scene
    payload = {
        "label": label,
        "target_area": round(room.area, 1),
        "area_ratio": round(room.area / max(scene.room.area, 1e-6), 3),
        "seconds": round(time.time() - t0, 1),
        "ours": {"topdown": topdown_png(ours),
                 "metrics": metrics_payload(graph, ours),
                 "n_objects": len([o for o in ours.objects if o.keep]),
                 "log": (res.info.get("summarization", [])
                         + res.info.get("population", []))[:8]},
    }
    if body.get("baseline", True):
        base = run_baseline("direct_scaling", graph, room, cfg=cfg)
        payload["baseline"] = {"topdown": topdown_png(base),
                               "metrics": metrics_payload(graph, base),
                               "n_objects": len(base.objects)}

    sid = str(body.get("session") or "default")
    with _LOCK:
        _SESSIONS[sid] = {"case": case_id, "room": room, "scene": ours,
                          "graph": graph, "reference": scene, "ts": time.time()}
        for k in [k for k, v in _SESSIONS.items()
                  if time.time() - v["ts"] > 7200]:
            _SESSIONS.pop(k, None)
    payload["session"] = sid
    payload["ours"]["objects"] = _object_list(ours)

    if assets is not None and body.get("render", True):
        rw, rh = (760, 540) if q == "good" else (640, 450)
        try:
            ra = repose_assets(assets, scene, ours)
            _, r1 = render_scene_textured(ra, room, rw, rh)
            payload["ours"]["render"] = _img_array(r1.rgb)
            if body.get("baseline", True):
                rb = repose_assets(assets, scene, base)
                _, r2 = render_scene_textured(rb, room, rw, rh)
                payload["baseline"]["render"] = _img_array(r2.rgb)
        except Exception as exc:
            print(f"  [render] {case_id}: {exc}", flush=True)
    return jsonify(payload)


def _graph_with_pins(graph, pins: set):
    """A copy of the reference graph with the named objects pinned."""
    import copy

    g = copy.copy(graph)
    g.scene = graph.scene.copy()
    for o in g.scene.objects:
        o.locked = o.oid in pins
    return g


def _keepout_polygons(room: Room, spec) -> list:
    """Rectangles painted on the plan, in fractions of the room's bounds.

    The page works in a normalised 0..1 box because it draws on the same
    top-down image the solver's plan comes from; converting here keeps the
    client from needing to know anything about metres.
    """
    b = room.bbox
    ext = np.maximum(b[1] - b[0], 1e-6)
    out = []
    for z in spec or []:
        try:
            x0, y0, x1, y1 = (float(z["x0"]), float(z["y0"]),
                              float(z["x1"]), float(z["y1"]))
        except (KeyError, TypeError, ValueError):
            continue
        if abs(x1 - x0) < 0.02 or abs(y1 - y0) < 0.02:
            continue
        lo = b[0] + np.array([min(x0, x1), min(y0, y1)]) * ext
        hi = b[0] + np.array([max(x0, x1), max(y0, y1)]) * ext
        out.append(np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                             [hi[0], hi[1]], [lo[0], hi[1]]]))
    return out


def _object_list(scene: Scene) -> list:
    return [{"oid": o.oid, "category": o.category,
             "x": round(float(o.xy[0]), 3), "y": round(float(o.xy[1]), 3),
             "z": round(float(o.z), 3),
             "yaw": round(float(o.yaw), 4),
             "size": [round(float(v), 3) for v in o.size],
             "jid": o.jid, "locked": bool(o.locked),
             "added": bool(o.meta.get("added")),
             "substituted": bool(o.meta.get("substituted_from"))}
            for o in scene.objects if o.keep]


@app.post("/api/edit")
def api_edit():
    """Apply hand edits to the last result: move, turn, delete, restore.

    This is the sense in which the output is editable -- every object is still
    an instance with its own pose and asset, so a person can overrule the
    solver on any one of them and the rest stays put.  Metrics are recomputed
    on the edited scene, so the cost of an edit is visible immediately.
    """
    body = request.get_json(force=True)
    sid = str(body.get("session") or "default")
    with _LOCK:
        st = _SESSIONS.get(sid)
    if st is None:
        return jsonify({"error": "no scene to edit — run a retarget first"}), 400
    scene, room, graph = st["scene"], st["room"], st["graph"]
    by = {o.oid: o for o in scene.objects}
    for e in body.get("edits", []):
        o = by.get(e.get("oid"))
        if o is None:
            continue
        act = e.get("op")
        if act == "move":
            o.position[0] = float(e["x"])
            o.position[1] = float(e["y"])
        elif act == "rotate":
            o.yaw = float(e["yaw"])
        elif act == "nudge":
            o.position[0] += float(e.get("dx", 0.0))
            o.position[1] += float(e.get("dy", 0.0))
            o.yaw += float(e.get("dyaw", 0.0))
        elif act == "delete":
            o.keep = False
        elif act == "restore":
            o.keep = True
        elif act == "pin":
            # C_t of section 1: from here on the solver may not move or drop it
            o.locked = bool(e.get("value", True))
    out = {"topdown": topdown_png(scene),
           "metrics": metrics_payload(graph, scene),
           "objects": _object_list(scene),
           "n_objects": len([o for o in scene.objects if o.keep])}
    if body.get("render", True):
        assets = _ASSETS.get(st["case"])
        if assets is not None:
            try:
                ra = repose_assets(assets, st["reference"], scene)
                _, r = render_scene_textured(ra, room, 640, 450)
                out["render"] = _img_array(r.rgb)
            except Exception as exc:
                print(f"  [render] edit: {exc}", flush=True)
    return jsonify(out)


@app.get("/api/export")
def api_export():
    """Download the current scene as JSON — poses, sizes and asset ids."""
    sid = str(request.args.get("session") or "default")
    with _LOCK:
        st = _SESSIONS.get(sid)
    if st is None:
        return jsonify({"error": "nothing to export"}), 400
    scene = st["scene"]
    d = scene.to_dict()
    d["objects"] = [o for o in d["objects"] if o.get("keep", True)]
    d["meta"] = dict(d.get("meta", {}))
    d["meta"]["exported"] = time.time()
    d["meta"]["reference"] = st["case"]
    from flask import Response
    return Response(json.dumps(d, indent=1),
                    mimetype="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="{st["case"][:24]}_scene.json"'})


@app.post("/api/vote")
def api_vote():
    body = request.get_json(force=True)
    body.pop("k", None)
    body["ts"] = time.time()
    path = CFG["votes"]
    if os.path.exists(path) and os.path.getsize(path) > 4_000_000:
        return jsonify({"error": "vote log full"}), 507
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(body) + "\n")
    n = sum(1 for _ in open(path))
    return jsonify({"ok": True, "n_votes": n})


@app.get("/api/votes")
def api_votes():
    path = CFG["votes"]
    if not os.path.exists(path):
        return jsonify({"n": 0, "preservation": {}, "suitability": {}})
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {"n": len(rows), "preservation": Counter(), "suitability": Counter()}
    for r in rows:
        out["preservation"][r.get("q1", "?")] += 1
        out["suitability"][r.get("q2", "?")] += 1
    out["preservation"] = dict(out["preservation"])
    out["suitability"] = dict(out["suitability"])
    return jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="outputs/references")
    ap.add_argument("--front", default="/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT")
    ap.add_argument("--future", nargs="+",
                    default=sorted(glob.glob(
                        "/home/gino/data/reroom/3D-FUTURE/3D-FUTURE-model-part*")))
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--votes", default="outputs/webapp_votes.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token", default=None,
                    help="require ?k=TOKEN; generated if omitted with --public")
    ap.add_argument("--public", action="store_true",
                    help="intended for exposure through a tunnel: forces a token")
    ap.add_argument("--rate", type=int, default=20,
                    help="max retargets per IP per minute")
    ap.add_argument("--device", default="cpu",
                    help="cpu is faster here; see the note at the top of the file")
    a = ap.parse_args()
    token = a.token
    if a.public and not token:
        import secrets
        token = secrets.token_urlsafe(12)

    CFG.update(refs=a.refs, front=a.front, future=a.future, votes=a.votes,
               token=token, rate=a.rate, device=a.device)
    CFG["elasticity"] = (load_elasticity(a.elasticity)
                         if os.path.exists(a.elasticity) else None)
    CFG["bank"] = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    if os.path.exists(a.cooc):
        d = json.load(open(a.cooc))
        CFG["cooc"] = CooccurrenceModel(
            counts={k: Counter(v) for k, v in d["counts"].items()},
            sizes={k: np.asarray(v) for k, v in d["sizes"].items()},
            n_scenes=d["n_scenes"])
    n = len(glob.glob(os.path.join(a.refs, "*", "meta.json")))
    print(f"ReRoom bench — {n} reference rooms, "
          f"{len(CFG['bank']) if CFG['bank'] else 0} assets")
    url = f"http://{a.host}:{a.port}"
    if token:
        url += f"/?k={token}"
        print(f"access key: {token}")
    print(f"open  {url}")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
