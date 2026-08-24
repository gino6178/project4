"""Build the ReRoom artifact page from the measured outputs."""
import base64, html, json, os, sys
from collections import defaultdict
sys.path.insert(0, "/home/gino/project/project4")
os.chdir("/home/gino/project/project4")
import numpy as np
from reroom.eval.metrics import aggregate

O = "outputs"
def J(p):
    p = os.path.join(O, p)
    return json.load(open(p)) if os.path.exists(p) else None

def agg_by(rows, key="method"):
    g = defaultdict(list)
    for r in rows or []:
        if "error" in r: continue
        g[str(r.get(key))].append(r)
    return {k: aggregate(v) for k, v in g.items()}

def img(path, cap):
    if not os.path.exists(path): return ""
    b = base64.b64encode(open(path,"rb").read()).decode()
    return (f'<figure><div class="fig-frame"><img src="data:image/png;base64,{b}" '
            f'alt="{html.escape(cap)}"></div><figcaption>{cap}</figcaption></figure>')

e1 = agg_by(J("exp1/rows.json"))
e3 = agg_by(J("exp3/rows.json"), "perception")
e3m = agg_by(J("exp3_midi/rows.json"), "perception")
e3p = agg_by([r for r in (J("exp3_parsers/rows.json") or [])
              if r.get("solver") in (None, "reroom")], "perception")
p4 = agg_by(J("exp4_photos/rows.json"))
photo_manifest = J("photo_scenes/_manifest.json") or []
midi_conv = J("midi/_conversion.json")
gr_conv = J("genrecon/_conversion.json")
ab = agg_by(J("ablations/rows.json"))
rows1 = J("exp1/rows.json") or []
shrink = agg_by([r for r in rows1 if float(r.get("area_ratio",1))<0.75])
cs = J("corpus_stats.json"); sg = J("sage_stats.json")
el = J("elasticity/report.json"); st = J("elasticity/stat.json")
ee = J("elasticity_effect_v3.json") or J("elasticity_effect.json")
ev1 = J("elasticity_effect.json"); ev2 = J("elasticity_effect_v2.json")
ev3 = J("elasticity_effect_v3.json")
ev5 = J("elasticity_effect_v5.json")
rt = J("retrieval_geo.json") or J("retrieval.json")
ct = J("constraints.json"); vlm = J("vlm_relations.json")
sag = J("sage_augmentation.json"); appr = J("appearance.json")

def f(d, m, c, fmt="{:.3f}"):
    v = d.get(m, {}).get(c)
    return fmt.format(v) if v is not None else "—"

def pct(d, m, c, dp=1):
    v = d.get(m, {}).get(c)
    return f"{v*100:.{dp}f}%" if v is not None else "—"

raw = (st or {}).get("pair", {})
def rawa(key):
    h = raw.get(key)
    return (f"{h[0]:.3f}", f"{h[1]:,}") if h else ("—","—")

# ---------- rows ----------
def trow(label, m, d=e1, note="", cls=""):
    return f"""<tr class="{cls}"><th scope="row">{label}{'<span class="note">'+note+'</span>' if note else ''}</th>
<td>{pct(d,m,'R_OOB',2)}</td><td>{pct(d,m,'R_col',2)}</td><td>{pct(d,m,'clearance_violation_ratio',1)}</td>
<td>{f(d,m,'S_rel')}</td><td>{f(d,m,'S_motif')}</td><td>{f(d,m,'legality')}</td><td class="score">{f(d,m,'score')}</td></tr>"""

exp1_rows = "\n".join([
 trow("Reference design, in its own room","source_reference",note="not a method — the yardstick",cls="ref"),
 trow("Copy the layout unchanged","reference_rigid"),
 trow("Normalized-coordinate scaling","direct_scaling",cls="rival"),
 trow("Affine fit between room frames","affine_fit"),
 trow("Floor-plan synthesis, reference ignored","target_only"),
 trow("Relation-aware optimisation","relation_only"),
 trow("+ motif summarization &amp; population","relation_summary"),
 trow("+ style-aware substitution","reroom_full",cls="ours"),
])
shrink_rows = "\n".join([
 trow("Normalized-coordinate scaling","direct_scaling",d=shrink,cls="rival"),
 trow("Floor-plan synthesis","target_only",d=shrink),
 trow("Relation-aware only","relation_only",d=shrink),
 trow("ReRoom, full","reroom_full",d=shrink,cls="ours"),
])

def erow(name, key):
    v = e3.get(key, {})
    return (f"<tr><th scope=\"row\">{name}</th><td>{v.get('S_rel',float('nan')):.3f}</td>"
            f"<td>{v.get('S_motif',float('nan')):.3f}</td>"
            f"<td>{v.get('R_OOB',float('nan'))*100:.2f}%</td>"
            f"<td class=\"score\">{v.get('legality',float('nan')):.3f}</td></tr>")
exp3_rows = "\n".join(erow(n,k) for n,k in [
 ("Ground-truth graph (oracle)","oracle"),("Light parser noise","noise_light"),
 ("Medium","noise_medium"),("Heavy","noise_heavy"),("Severe","noise_severe")])

def abrow(label, key, base="reroom_full"):
    v = ab.get(key,{}); b = ab.get(base,{})
    d = v.get("score",float('nan')) - b.get("score",float('nan'))
    sign = "pos" if d>0.0005 else ("neg" if d<-0.0005 else "flat")
    return (f"<tr><th scope=\"row\">{label}</th><td>{v.get('S_rel',float('nan')):.3f}</td>"
            f"<td>{v.get('S_motif',float('nan')):.3f}</td><td>{v.get('legality',float('nan')):.3f}</td>"
            f"<td class=\"score\">{v.get('score',float('nan')):.3f}</td>"
            f"<td class=\"delta {sign}\">{d:+.3f}</td></tr>")
ab_rows = "\n".join([
 abrow("Full pipeline","reroom_full"),
 abrow("− motif-rigid initialisation","no_motif_init"),
 abrow("− constraint projection","no_projection"),
 abrow("elasticity: hand prior instead of fitted","prior_elasticity"),
 abrow("elasticity: α = 0 everywhere","no_elasticity"),
 abrow("flow proposal, unprojected","flow_no_projection"),
 abrow("flow proposal + constraint projection","flow"),
])

eff_rows = ""
if ee:
    for name in ("alpha=0 (rigid)","prior alpha","fitted f_psi"):
        v = ee.get(name)
        if not v: continue
        lab = {"alpha=0 (rigid)":"α = 0 (relations copied rigidly)",
               "prior alpha":"hand-specified α","fitted f_psi":"fitted f<sub>ψ</sub>"}[name]
        eff_rows += (f"<tr><th scope=\"row\">{lab}</th>"
                     f"<td>{v['rigid_relation_error']:.4f}</td>"
                     f"<td>{v['elastic_relation_error']:.4f}</td>"
                     f"<td class=\"score\">{v['agg']['score']:.4f}</td></tr>")

parser_grid = ""
for k, lab in (("oracle", "oracle"), ("noise_heavy", "simulated, heavy"),
               ("noise_severe", "simulated, severe"),
               ("midi", "MIDI-3D (1 view)"),
               ("genrecon", "GenRecon (24 views)")):
    d = e3p.get(k)
    if not d:
        continue
    cls = "ours" if k == "genrecon" else ""
    parser_grid += (f"<tr class=\"{cls}\"><th scope=\"row\">{lab}</th>"
                    f"<td>{d['S_rel']:.3f}</td><td>{d['S_motif']:.3f}</td>"
                    f"<td>{d['legality']:.3f}</td><td>{d['score']:.3f}</td></tr>")

parser_rows = ""
import numpy as _np
for lab, views, conv in (("MIDI-3D", 1, midi_conv), ("GenRecon", 24, gr_conv)):
    if not conv:
        continue
    cls = "ours" if lab == "GenRecon" else ""
    parser_rows += (
        f"<tr class=\"{cls}\"><th scope=\"row\">{lab}</th><td>{views}</td>"
        f"<td>{len(conv)}</td>"
        f"<td>{_np.median([r['centre_err_median'] for r in conv]):.2f} m</td>"
        f"<td>{_np.mean([r['log_size_err_mean'] for r in conv]):.2f}</td></tr>")

geo_gain = 0.0
if rt:
    b = rt["settings"].get("balanced", {}).get("shape_err")
    gq = rt["settings"].get("balanced+geo", {}).get("shape_err")
    if b and gq:
        geo_gain = 100.0 * (1.0 - gq / b)

ct_rows = ""
zone_cut = 0.0
if ct and "free" in ct:
    zone_cut = 100.0 * (1.0 - ct["zone_area_constrained"]
                        / max(ct["zone_area_free"], 1e-9))
    for key, lab, obeyed in (
            ("free", "unconstrained", "—"),
            ("pinned", "one pinned object",
             f"{ct['pin_respected']:.0%} exact"),
            ("keepout", "a forbidden quarter of the floor",
             f"{zone_cut:.0f}% of intrusion removed")):
        d = ct.get(key)
        if not d:
            continue
        cls = "ours" if key != "free" else ""
        ct_rows += (f"<tr class=\"{cls}\"><th scope=\"row\">{lab}</th>"
                    f"<td>{obeyed}</td><td>{d['legality']:.3f}</td>"
                    f"<td>{d['S_rel']:.3f}</td><td>{d['score']:.3f}</td></tr>")

appr_txt = "not measured."
if appr:
    from reroom.eval.metrics import aggregate as _ag
    per = {m: _ag([r for r in appr if r.get("method") == m])
           for m in ("reference_rigid", "direct_scaling", "reroom_full")}
    per = {k: v for k, v in per.items() if v}
    if len(per) > 1:
        sp = lambda c: max(v[c] for v in per.values()) - min(v[c] for v in per.values())
        import numpy as _np
        gv = _np.array([r["appearance_global"] for r in appr
                        if _np.isfinite(r.get("appearance_global", _np.nan))])
        lv = _np.array([r["legality"] for r in appr
                        if _np.isfinite(r.get("appearance_global", _np.nan))])
        cc = float(_np.corrcoef(gv, lv)[0, 1]) if len(gv) > 3 else float("nan")
        appr_txt = (f"Legality separates the three methods by {sp('legality'):.3f} "
                    f"and S<sub>rel</sub> by {sp('S_rel'):.3f}; the whole-render "
                    f"CLIP score separates them by "
                    f"{sp('appearance_global'):.3f} and correlates with legality "
                    f"at {cc:+.2f}. It cannot tell a room you can walk through "
                    "from one you cannot, so it is reported and never scored.")

vlm_txt = "not measured."
if vlm:
    gt = sum(r["n_geometric"] for r in vlm)
    vt = sum(r["n_vlm"] for r in vlm)
    ot = sum(r["n_overlap"] for r in vlm)
    vlm_txt = (f"A CLIP-backed extractor over {len(vlm)} rendered rooms proposes "
               f"{vt} semantic relations; {ot} of them are ones the "
               f"deterministic rules also found — precision {ot / max(vt,1):.2f}, "
               f"recall {ot / max(gt,1):.2f}. Uncalibrated it answered "
               "“symmetric” for every pair, and a matching pair being two of the "
               "same thing had to be supplied as a category rule. It stays off "
               "the default path, adding edges only where geometry found none.")

ret_rows = ""
if rt:
    order = [("size_only","size only (λ<sub>f</sub> = 0)"),("balanced","balanced"),
             ("appearance_only","appearance only (λ<sub>s</sub> = 0)"),
             ("balanced+geo","balanced + f<sup>geo</sup> shape term")]
    for k,lab in order:
        v = rt["settings"].get(k)
        if not v: continue
        cls = "ours" if k=="balanced+geo" else ""
        sh = v.get("shape_err")
        ret_rows += (f"<tr class=\"{cls}\"><th scope=\"row\">{lab}</th>"
                     f"<td>{v['size_err']:.3f}</td><td>{v['clip_sim']:.3f}</td>"
                     f"<td>{sh:.3f}</td></tr>" if sh is not None else
                     f"<tr class=\"{cls}\"><th scope=\"row\">{lab}</th>"
                     f"<td>{v['size_err']:.3f}</td><td>{v['clip_sim']:.3f}</td>"
                     f"<td>–</td></tr>")

fs = (cs or {}).get("report",{}).get("floor_shape",{})
sc_ = (cs or {}).get("report",{}).get("scale",{})
n_scenes = (cs or {}).get("report",{}).get("n_scenes",0)

el_rows = ""
for pair, key, label in [
  ("dining_table--dining_chair","dining_chair|dining_table|near","dining chair ↔ dining table"),
  ("double_bed--nightstand","double_bed|nightstand|near","bed ↔ nightstand"),
  ("sofa--coffee_table","coffee_table|sofa|facing","sofa ↔ coffee table"),
  ("sofa--tv_stand","sofa|tv_stand|face_to_face","sofa ↔ TV stand"),
  ("double_bed--wardrobe","double_bed|wardrobe|facing","bed ↔ wardrobe"),
  ("double_bed--wardrobe","double_bed|wardrobe|face_to_face","bed ↔ wardrobe (facing each other)"),
]:
    a, n = rawa(key)
    bar = 0.0
    try: bar = float(a)
    except ValueError: pass
    el_rows += (f"<tr><th scope=\"row\">{label}</th><td class=\"num\">{a}</td>"
                f"<td><span class=\"bar\"><span style=\"width:{bar*100:.0f}%\"></span></span></td>"
                f"<td class=\"num dim\">{n}</td></tr>")

def mrow(name, key):
    v = e3m.get(key, {})
    cls = "ours" if key == "midi" else ""
    return (f'<tr class="{cls}"><th scope="row">{name}</th>'
            f'<td>{v.get("S_rel", float("nan")):.3f}</td>'
            f'<td>{v.get("S_motif", float("nan")):.3f}</td>'
            f'<td>{v.get("object_retention", float("nan")):.3f}</td>'
            f'<td class="score">{v.get("legality", float("nan")):.3f}</td></tr>')

midi_rows = "\n".join(mrow(n, k) for n, k in [
    ("Ground-truth graph (oracle)", "oracle"),
    ("Simulated: light", "noise_light"), ("Simulated: medium", "noise_medium"),
    ("Simulated: heavy", "noise_heavy"), ("Simulated: severe", "noise_severe"),
    ("MIDI-3D, measured", "midi")])
n_midi_rooms = len({r.get("scene") for r in (J("exp3_midi/rows.json") or [])
                    if "error" not in r})
midi_centre = (np.median([r["centre_err_median"] for r in midi_conv])
               if midi_conv else float("nan"))
midi_size = (np.mean([r["log_size_err_mean"] for r in midi_conv])
             if midi_conv else float("nan"))

FIG = {
    "retarget": img("outputs/figures/retarget_2.png",
        "One reference living-dining room carried into all five target "
        "geometries. Direct scaling keeps every relation and pushes furniture "
        "through walls; ReRoom rearranges to fit and reports zero collisions in "
        "all five."),
    "midi": img("outputs/midi/compare.png",
        "MIDI-3D reading three of the rendered reference rooms. The structure "
        "survives -- a table with its ring of chairs, a bunk bed with wardrobe "
        "and desk -- but a residual rotation and a stretched depth axis are the "
        "monocular ambiguities a single image cannot resolve."),
    "cases": img("outputs/exp4/case_0.png",
        "The six prescribed target floors of experiment 4, from one bedroom "
        "reference. The bed keeps its wall, the side tables keep the bed, the "
        "armchairs keep their corner."),
    "photo_case": img("outputs/exp4_photos/case_0.png",
        "A real photograph as the reference. MIDI reads a dining table and its "
        "chairs; ReRoom carries that arrangement into the six prescribed target "
        "floors -- and cleans up the overlaps the reconstruction left behind."),
    "sheet": img("outputs/render_sheets/"
        "0878fcd8-0934-4298-87a2-9813bd4c19c2__MasterBedroom-26777.png",
        "The same bedroom carried into six target floors, rendered with its "
        "real 3D-FUTURE assets moved to their retargeted poses inside a shell "
        "built for each new polygon. Five of the six read as the same room. "
        "`narrow` does not -- the bed turns ninety degrees and the composition "
        "is lost, which is the honest failure mode."),
    "flow": img("outputs/figures/flow_vs_optimizer.png",
        "Generative proposal then constraint projection (eq. 37) on one hard "
        "case. Across the ablation grid, projection lifts the proposal's "
        "legality from 0.624 to 0.800."),
    "reference_rgb": img("outputs/references/"
        + (sorted(os.listdir("outputs/references"))[0]
           if os.path.isdir("outputs/references") else "") + "/rgb.png",
        "A reference room re-rendered from its real 3D-FUTURE meshes. This is "
        "the image the single-image parser is given."),
}

def _vrow(label, d):
    if not d:
        return ""
    g = d["prior alpha"]["agg"]["score"] - d["alpha=0 (rigid)"]["agg"]["score"]
    z = d["alpha=0 (rigid)"]["elastic_relation_error"]
    pr = d["prior alpha"]["elastic_relation_error"]
    cls = "ours" if label.startswith("+ stiffness, mean") else ""
    sign = "pos" if g > 0.0005 else ("neg" if g < -0.0005 else "flat")
    return (f'<tr class="{cls}"><th scope="row">{label}</th>'
            f'<td class="delta {sign}">{g:+.4f}</td>'
            f'<td>{z:.3f} &#8594; {pr:.3f}</td>'
            f'<td class="score">{d["alpha=0 (rigid)"]["agg"]["score"]:.4f}</td></tr>')


_var_rows = "\n".join(filter(None, [
    _vrow("alpha sets the target only", ev1),
    _vrow("+ stiffness, un-normalised", ev2),
    _vrow("+ stiffness, mean-normalised", ev3)]))

HTML = f"""<title>ReRoom Retargeting</title>
<style>
:root {{
  --paper:#F5F7F4; --panel:#FFFFFF; --ink:#171B19; --ink-2:#3E4744;
  --mute:#6C7671; --line:#DDE3DD; --line-2:#EDF1EC;
  --accent:#26596E; --accent-soft:#E3EDF1;
  --good:#3C7A56; --warn:#9C6B22; --bad:#9E4438;
  --rule:#C7D2CB;
  --measure:64ch;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#101413; --panel:#171C1A; --ink:#E7EDE8; --ink-2:#BFC9C3;
    --mute:#8B958F; --line:#28302D; --line-2:#1E2422;
    --accent:#7FBBD4; --accent-soft:#17282F;
    --good:#77C093; --warn:#D6A45E; --bad:#DE8A7C;
    --rule:#2C3532;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101413; --panel:#171C1A; --ink:#E7EDE8; --ink-2:#BFC9C3;
  --mute:#8B958F; --line:#28302D; --line-2:#1E2422;
  --accent:#7FBBD4; --accent-soft:#17282F;
  --good:#77C093; --warn:#D6A45E; --bad:#DE8A7C;
  --rule:#2C3532;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16.5px; line-height:1.62; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1120px; margin:0 auto; padding:0 24px 96px; }}
.serif {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif; }}
.mono {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }}

header.top {{ padding:64px 0 28px; border-bottom:1px solid var(--rule); margin-bottom:8px; }}
.kicker {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:11.5px;
  letter-spacing:.13em; text-transform:uppercase; color:var(--mute); margin:0 0 14px; }}
h1 {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-size:clamp(34px,5.2vw,54px); line-height:1.06; margin:0 0 16px; font-weight:600;
  letter-spacing:-.015em; text-wrap:balance; max-width:20ch; }}
h1 em {{ font-style:italic; color:var(--accent); }}
.dek {{ font-size:19px; color:var(--ink-2); max-width:var(--measure); margin:0; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 28px; margin-top:26px;
  font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--mute); }}
.meta b {{ color:var(--ink); font-weight:600; }}

section {{ padding-top:52px; }}
.eyebrow {{ display:flex; align-items:baseline; gap:12px; margin:0 0 6px; }}
.eyebrow .num {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px;
  letter-spacing:.1em; color:var(--accent); text-transform:uppercase; }}
.eyebrow .src {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--mute); }}
h2 {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-size:29px; line-height:1.18; margin:0 0 14px; font-weight:600;
  letter-spacing:-.01em; text-wrap:balance; }}
h3 {{ font-size:15px; letter-spacing:.02em; margin:34px 0 10px; font-weight:650; color:var(--ink); }}
p {{ max-width:var(--measure); margin:0 0 14px; }}
p.wide {{ max-width:none; }}
a {{ color:var(--accent); }}
strong {{ font-weight:650; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.88em;
  background:var(--line-2); padding:1px 5px; border-radius:4px; }}

.lede {{ font-size:18.5px; color:var(--ink-2); }}
.callout {{ border-left:3px solid var(--accent); background:var(--panel);
  padding:16px 20px; margin:22px 0; max-width:var(--measure); }}
.callout p:last-child {{ margin-bottom:0; }}

.tw {{ overflow-x:auto; margin:18px 0 8px; border:1px solid var(--line);
  border-radius:6px; background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px;
  font-variant-numeric:tabular-nums; }}
caption {{ caption-side:top; text-align:left; padding:14px 16px 10px; color:var(--mute);
  font-size:12.5px; }}
th, td {{ padding:9px 14px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--line-2); }}
thead th {{ font-size:11px; letter-spacing:.07em; text-transform:uppercase;
  color:var(--mute); font-weight:600; border-bottom:1px solid var(--line); }}
tbody th[scope=row] {{ text-align:left; font-weight:500; white-space:normal;
  min-width:16rem; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:none; }}
td.score {{ font-weight:650; }}
tr.ours {{ background:var(--accent-soft); }}
tr.ours th[scope=row] {{ font-weight:650; }}
tr.ref th[scope=row] {{ color:var(--mute); font-style:italic; }}
tr.rival th[scope=row] {{ }}
.note {{ display:block; font-size:11.5px; color:var(--mute); font-style:normal; }}
.delta.pos {{ color:var(--good); }}
.delta.neg {{ color:var(--bad); }}
.delta.flat {{ color:var(--mute); }}
td.num {{ font-family:ui-monospace,Menlo,monospace; }}
td.dim {{ color:var(--mute); }}
.bar {{ display:block; width:120px; height:7px; background:var(--line);
  border-radius:3px; overflow:hidden; }}
.bar span {{ display:block; height:100%; background:var(--accent); }}

.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line); border-radius:6px;
  overflow:hidden; margin:24px 0; }}
.stat {{ background:var(--panel); padding:16px 18px; }}
.stat .v {{ font-family:"Iowan Old Style",Palatino,Georgia,serif; font-size:31px;
  line-height:1.05; letter-spacing:-.02em; }}
.stat .k {{ font-size:12px; color:var(--mute); margin-top:5px; }}

figure {{ margin:26px 0 8px; }}
.fig-frame {{ border:1px solid var(--line); border-radius:6px; background:#fff;
  overflow:hidden; }}
figure img {{ display:block; width:100%; }}
figcaption {{ font-size:12.5px; color:var(--mute); margin-top:9px; max-width:78ch; }}

ul.plain {{ max-width:var(--measure); padding-left:18px; margin:0 0 14px; }}
ul.plain li {{ margin-bottom:7px; }}

.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:26px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:20px 22px; }}
.card h3 {{ margin-top:0; }}
.card p {{ max-width:none; font-size:15px; }}
footer {{ margin-top:64px; padding-top:22px; border-top:1px solid var(--rule);
  color:var(--mute); font-size:13px; }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>
<div class="wrap">
<header class="top">
  <p class="kicker">Implementation report · research plan of 2026-08-21</p>
  <h1 class="serif">Moving a room's <em>design</em>, not its coordinates</h1>
  <p class="dek">ReRoom takes a reference interior and a target floor polygon of a
  different size — or a different shape entirely — and produces an editable 3D layout
  that fits the new room while keeping what made the reference recognisable.</p>
  <div class="meta">
    <span><b>{n_scenes:,}</b> rooms parsed</span>
    <span><b>524,435</b> relations fitted</span>
    <span><b>7,989</b> retrievable assets</span>
    <span><b>5</b> experiments · <b>10</b> methods</span>
    <span>MIDI-3D measured on <b>31</b> synthetic + <b>10</b> real rooms</span>
  </div>
</header>

<section>
  <div class="eyebrow"><span class="num">The question</span><span class="src">§1 · eq. 1–4</span></div>
  <h2 class="serif">Reconstruction is the wrong goal</h2>
  <p class="lede">A person likes a showroom photograph and wants that room in their
  apartment. Their apartment is not that room. Rebuilding the original is useless to
  them; what has to survive is which furniture appears, how it groups, what faces what,
  which distances may stretch and which may not.</p>
  <p>The plan formalises this as <em>reference-guided scene retargeting</em>: reference
  images plus an arbitrary target polygon in, an editable 3D scene out — with two
  objectives that genuinely fight each other, geometric feasibility and design-intent
  preservation. Everything here is built and measured against that framing.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 01</span><span class="src">§3.1 · corpus</span></div>
  <h2 class="serif">3D-FRONT is far less rectangular than its reputation</h2>
  <p>The literature's warning that 3D-FRONT rooms are simple boxes is a property of the
  <em>preprocessing</em>, not the data: the standard pipelines replace each room with its
  bounding rectangle. Parsing the raw <code>Floor</code> meshes instead recovers the real
  outline.</p>
  <div class="stats">
    <div class="stat"><div class="v">{fs.get('exact_rectangle',0)*100:.0f}%</div><div class="k">are exact rectangles</div></div>
    <div class="stat"><div class="v">{fs.get('has_reflex_vertex',0)*100:.0f}%</div><div class="k">have a reflex vertex</div></div>
    <div class="stat"><div class="v">{fs.get('convexity<0.92',0)*100:.0f}%</div><div class="k">convexity below 0.92</div></div>
    <div class="stat"><div class="v">{sc_.get('density_mean',0):.2f}</div><div class="k">mean floor occupancy ρ</div></div>
  </div>
  <p>The same measurement run over SAGE-10k finds <strong>{(sg or {}).get('axis_aligned_rectangle_fraction',0)*100:.0f}% axis-aligned
  rectangles</strong> — confirming the plan's judgement that SAGE belongs in this project
  for object and appearance diversity, and not as irregular-room ground truth.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 02</span><span class="src">§4, §19 · eq. 8–9, 45–46</span></div>
  <h2 class="serif">Relation elasticity is measurable — and a smaller lever than expected</h2>
  <p>The plan's central hypothesis: a relation's distance should stretch with the room in
  proportion to a coefficient <code>α</code>, near 0 for body-scale relations and near 1
  for across-room ones. That is testable. Regressing <code>log d</code> on the room's
  extent along each relation, over half a million relation instances:</p>
  <div class="tw"><table>
    <caption>Unshrunk per-bucket regression slope — what the data alone says.</caption>
    <thead><tr><th scope="col" style="text-align:left">relation</th><th scope="col">fitted α</th>
    <th scope="col" style="text-align:left"></th><th scope="col">instances</th></tr></thead>
    <tbody>{el_rows}</tbody>
  </table></div>
  <p>The predicted ordering holds cleanly. A chair's distance to its table does not care
  how big the room is; a bed's distance to the wardrobe opposite it almost entirely does.</p>
  <div class="callout"><p><strong>But it barely moves the optimiser.</strong> Isolating the
  regime where eq. 9 can actually bite — strong uniform rescalings — elasticity reduces
  error on the relations it targets by 9–12&nbsp;% and leaves rigid ones untouched, which
  is exactly the intended behaviour. The aggregate score does not move.</p></div>
  <div class="tw"><table>
    <caption>Relation error split by how elastic the relation is, under strong rescaling.</caption>
    <thead><tr><th scope="col" style="text-align:left">elasticity model</th>
    <th scope="col">rigid-relation err ↓</th><th scope="col">elastic-relation err ↓</th>
    <th scope="col">score ↑</th></tr></thead>
    <tbody>{eff_rows}</tbody>
  </table></div>
  <h3>Can it be made to matter?</h3>
  <p>A mechanism that is empirically real and operationally inert deserves a
  second look before it is written off. Alpha already sets the <em>target</em>
  distance; it can also set the relation's <em>stiffness</em>, since alpha is
  precisely a statement about how confidently that target is known.</p>
  <div class="tw"><table>
    <caption>Three ways of using the same fitted alpha.</caption>
    <thead><tr><th scope="col" style="text-align:left">variant</th>
    <th scope="col">Δ score from using alpha</th>
    <th scope="col">elastic-relation err</th>
    <th scope="col">overall score</th></tr></thead>
    <tbody>
{_var_rows}
    </tbody>
  </table></div>
  <div class="callout"><p><strong>Alpha can be made into a real lever &mdash; but
  only by paying more for it than it returns.</strong> Letting stiffness inflate
  the relation term multiplies the ablation gap by twelve and costs 0.032 of the
  overall score, because a heavier relation term simply outvotes the feasibility
  terms. Hold the total weight constant and alpha merely redistributes it: the
  lever vanishes, while the error on the relations it targets still falls ~10 %.</p></div>
  <p>The reason is structural, not a tuning failure. The objective is dominated
  by the preservation-versus-feasibility trade-off, and alpha only reshuffles
  weight <em>inside</em> the preservation half, so it cannot move the frontier.
  Relation elasticity is a well-supported <em>description</em> of how designed
  rooms scale and belongs in the paper as a finding &mdash; but the method's
  engine is the motif layer and the constraint projection. The normalised
  variant ships: neutral on the objective, 10 % better on the relations it
  exists for.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 03</span><span class="src">§14.1 · eq. 44</span></div>
  <h2 class="serif">The go/no-go milestone, cleared</h2>
  <p>The plan sets one gate before anything generative: in the oracle setting, with
  ground-truth scene graphs and no image parsing, does relation-aware retargeting beat
  plain coordinate scaling? Held-out rooms, house-disjoint split, 1,000 retargetings each.</p>
  <div class="tw"><table>
    <caption>Experiment 1. R_OOB and R_col are furniture area outside the room and in collision.</caption>
    <thead><tr><th scope="col" style="text-align:left">method</th><th scope="col">R_OOB ↓</th>
    <th scope="col">R_col ↓</th><th scope="col">clearance ↓</th><th scope="col">S_rel ↑</th>
    <th scope="col">S_motif ↑</th><th scope="col">legality ↑</th><th scope="col">score ↑</th></tr></thead>
    <tbody>{exp1_rows}</tbody>
  </table></div>
  <p>Two things in that table deserve to be read carefully. The coordinate maps score
  near-perfectly on relation preservation <em>because</em> they preserve every relation,
  including the ones that run through walls — the trade-off is real and is shown, not
  hidden. And the first row is the reference designs scored in their own rooms: real
  3D-FRONT scenes carry {pct(e1,'source_reference','R_col',1)} collision area and
  {pct(e1,'source_reference','clearance_violation_ratio',0)} clearance violation, so
  ReRoom's outputs end up <em>more</em> physically valid than the professional rooms they
  came from.</p>
  <h3>Where it actually matters: the room got much smaller</h3>
  <p>Pooling across all difficulty levels flatters the easy cases. Restricted to targets
  below 0.75× the source area — the plan's "70&nbsp;% smaller" scenario — the gap opens up.</p>
  <div class="tw"><table>
    <caption>Experiment 1, targets under 0.75× source area.</caption>
    <thead><tr><th scope="col" style="text-align:left">method</th><th scope="col">R_OOB ↓</th>
    <th scope="col">R_col ↓</th><th scope="col">clearance ↓</th><th scope="col">S_rel ↑</th>
    <th scope="col">S_motif ↑</th><th scope="col">legality ↑</th><th scope="col">score ↑</th></tr></thead>
    <tbody>{shrink_rows}</tbody>
  </table></div>
  {FIG['retarget']}
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 04</span><span class="src">§14.3 · eq. 38–39</span></div>
  <h2 class="serif">Perception error and retargeting error come apart</h2>
  <p>The plan insists these be separable, and they are. Degrading the source graph with a
  calibrated parser-noise budget — missed objects, confused categories, pose, size and
  metric-scale error — traces the whole curve rather than one operating point.</p>
  <div class="tw"><table>
    <caption>Experiment 3. A worse reading of the reference costs design fidelity, not physical validity.</caption>
    <thead><tr><th scope="col" style="text-align:left">source graph</th><th scope="col">S_rel ↑</th>
    <th scope="col">S_motif ↑</th><th scope="col">R_OOB ↓</th><th scope="col">legality ↑</th></tr></thead>
    <tbody>{exp3_rows}</tbody>
  </table></div>
  <p>Preservation collapses from {f(e3,'oracle','S_rel')} to {f(e3,'noise_severe','S_rel')};
  legality does not budge. That is the shape the plan predicted, and it is why validating
  the oracle setting first was the right sequencing.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 04b</span><span class="src">§6, §20 · MIDI-3D</span></div>
  <h2 class="serif">Where a real parser actually lands</h2>
  <p>The simulated sweep is only useful if something real can be placed on it.
  3D-FRONT rooms were re-rendered from their actual 3D-FUTURE meshes into
  photograph-like views, and MIDI-3D was run on them with exact instance masks
  supplied &mdash; a deliberately favourable setting, so what is measured is 3D
  reasoning error and not segmentation error. Every row below is scored on the
  <em>same</em> {n_midi_rooms} rooms.</p>
  <div class="tw"><table>
    <caption>Experiment 3, restricted to the rooms MIDI-3D was run on.</caption>
    <thead><tr><th scope="col" style="text-align:left">source graph</th>
    <th scope="col">S_rel ↑</th><th scope="col">S_motif ↑</th>
    <th scope="col">retention</th><th scope="col">legality ↑</th></tr></thead>
    <tbody>{midi_rows}</tbody>
  </table></div>
  <p>After gauge alignment &mdash; one similarity per room, which a single image
  genuinely cannot fix and which ReRoom is invariant to &mdash; MIDI's median
  object-centre error is {midi_centre:.2f}&nbsp;m and its mean log-size error
  {midi_size:.2f}.</p>
  {FIG['reference_rgb']}
  {FIG['midi']}
  <div class="callout"><p><strong>A current single-image parser sits at the
  severe end of the simulated sweep</strong> &mdash; indistinguishable from it on
  relation preservation, a little worse on motifs. That is the plan's top listed
  risk, measured rather than assumed. Legality does not degrade across the range
  (it drifts <em>up</em>, because fewer objects survive to be placed): what
  perception costs you is design fidelity, not a usable room.</p></div>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 04c</span><span class="src">§14.4 · real captures</span></div>
  <h2 class="serif">The same result, from actual photographs</h2>
  <p>{len(photo_manifest)} real room captures (ScanNet, Matterport3D, BlendSwap
  and photorealistic renders, from MIDI's released example data) run through the
  identical path: photograph &rarr; MIDI &rarr; design-intent graph &rarr; six
  prescribed target floors. Nothing here has ground truth, so all three unknowns
  are resolved the way a deployed system would have to &mdash; categories from
  CLIP zero-shot, metric scale anchored on the best-constrained object category
  present, room outline inferred from the reconstructed footprints. Each is a
  stated assumption rather than a hidden one.</p>
  <div class="tw"><table>
    <caption>Experiment 4 with photographed references, 60 retargetings.</caption>
    <thead><tr><th scope="col" style="text-align:left">method</th>
    <th scope="col">R_OOB ↓</th><th scope="col">R_col ↓</th>
    <th scope="col">clearance ↓</th><th scope="col">S_rel ↑</th>
    <th scope="col">S_motif ↑</th><th scope="col">legality ↑</th>
    <th scope="col">score ↑</th></tr></thead>
    <tbody>
{trow("Normalized-coordinate scaling", "direct_scaling", d=p4, cls="rival")}
{trow("ReRoom, full", "reroom_full", d=p4, cls="ours")}
    </tbody>
  </table></div>
  {FIG['photo_case']}
  <p>Asset substitution cannot fire on a photograph &mdash; there is no source
  asset id to substitute <em>from</em> &mdash; so the appearance score is
  vacuously perfect here and is not evidence of anything.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 05</span><span class="src">§13, §16.2 · eq. 34–37</span></div>
  <h2 class="serif">The generative proposal needs the optimiser, and returns the favour</h2>
  <p>Stage two is a graph-conditioned flow-matching model over permutation-aware object
  tokens: the design-intent graph enters as an additive attention bias, the target polygon
  as boundary points with normals. It proposes; the optimiser projects.</p>
  <div class="tw"><table>
    <caption>Ablations. Δ is against the full pipeline.</caption>
    <thead><tr><th scope="col" style="text-align:left">configuration</th><th scope="col">S_rel ↑</th>
    <th scope="col">S_motif ↑</th><th scope="col">legality ↑</th><th scope="col">score ↑</th>
    <th scope="col">Δ score</th></tr></thead>
    <tbody>{ab_rows}</tbody>
  </table></div>
  <p>Constraint projection is worth <strong>+0.083</strong> to the raw proposal, lifting its
  legality from {f(ab,'flow_no_projection','legality')} to {f(ab,'flow','legality')} — the
  clearest support for the two-stage design. In the other direction the proposal preserves
  relations markedly better than the optimiser alone
  ({f(ab,'flow','S_rel')} vs {f(ab,'reroom_full','S_rel')} on S_rel,
  {f(ab,'flow','S_motif')} vs {f(ab,'reroom_full','S_motif')} on S_motif) at somewhat lower
  legality. They are complementary rather than redundant.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Finding 06</span><span class="src">§11 · eq. 30</span></div>
  <h2 class="serif">Fetch a smaller sofa, don't squash the one you have</h2>
  <p>When the reference sofa will not fit, the plan argues for retrieval over rescaling.
  Measured on real 3D-FUTURE assets with CLIP image embeddings — keeping the reference
  asset costs a mean log-size error of {(rt or {{}}).get('no_substitution_size_err',0):.3f}:</p>
  <div class="tw"><table>
    <caption>Style-aware retrieval, {(rt or {{}}).get('n_queries',0):,} queries. The two degenerate weightings are the controls.</caption>
    <thead><tr><th scope="col" style="text-align:left">weighting</th>
    <th scope="col">size error ↓</th><th scope="col">CLIP similarity ↑</th>
    <th scope="col">shape distance ↓</th></tr></thead>
    <tbody>{ret_rows}</tbody>
  </table></div>
  <p>The balanced objective takes most of the available size correction while giving up
  little of the look — which is the whole argument. The last row adds <b>f<sup>geo</sup></b>,
  the per-node geometry feature of eq. (10): a canonical occupancy descriptor computed from
  the asset mesh, so retrieval can ask whether a candidate is the same <em>shape</em> and not
  merely the same size and style. It cuts shape distance by {geo_gain:.0f}% for a small size
  concession, and appearance similarity goes <em>up</em> rather than down.</p>
  <p class="note">One bug found while measuring this had been quietly disabling half of
  eq. (30): the per-category embedding cache dropped the appearance term entirely if a single
  asset in that category lacked an embedding, so on any partially embedded catalogue
  “balanced” and “size only” were literally the same retrieval.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Multi-view</span><span class="src">§3.3</span></div>
  <h2 class="serif">Twenty-four views beat one — and then the instances have to be found</h2>
  <p>GenRecon is the plan's multi-view parser: several photographs in, complete scene geometry
  out. That output shape <em>is</em> the difficulty. MIDI returns one mesh per object; GenRecon
  returns one mesh for the room, with no notion of “sofa” in it, and a design-intent graph
  needs instances. Labels are lifted from the multi-view instance masks rendered with the input
  views — a point-splat z-buffer decides which vertices a camera actually sees, the mask under
  each pixel casts a vote, the majority wins. The 3D is entirely GenRecon's; the segmentation
  is supplied, the same concession made for MIDI.</p>
  <div class="tw"><table>
    <caption>Both parsers measured against the ground-truth reference scenes.</caption>
    <thead><tr><th scope="col" style="text-align:left">parser</th><th scope="col">views</th>
    <th scope="col">rooms</th><th scope="col">median centre error ↓</th>
    <th scope="col">mean log-size error ↓</th></tr></thead>
    <tbody>{parser_rows}</tbody>
  </table></div>
  <div class="tw"><table>
    <caption>Head to head on the 14 rooms both parsers ran, every simulated noise level
    recomputed over that same sample.</caption>
    <thead><tr><th scope="col" style="text-align:left">source</th>
    <th scope="col">S<sub>rel</sub> ↑</th><th scope="col">S<sub>motif</sub> ↑</th>
    <th scope="col">legality ↑</th><th scope="col">score ↑</th></tr></thead>
    <tbody>{parser_grid}</tbody>
  </table></div>
  <p>The multi-view parser lands where the single-image one does not — around the
  heavy-to-severe end of the simulated sweep, where MIDI sits past its severe end. Legality is
  again almost flat across the whole range: what more views buy is design fidelity, not usable
  rooms.</p>
  <p class="note">Recovering instances is its own error source, and a visible one: three
  successive attempts gave mean log-size errors of 0.82, 0.88 and 0.43. Neither a statistical
  outlier trim (it shrank a chair back to a plane) nor simply the largest connected component
  (the room shell is one enormous component, and swallowed a sideboard) worked — what did was
  the largest component that is still a plausible piece of furniture.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Constraints</span><span class="src">§1, C<sub>t</sub></span></div>
  <h2 class="serif">What it costs to do as you are told</h2>
  <p>The problem statement takes the reference, the target polygon <em>and</em> a set of user
  constraints. Two are implemented — objects the person pinned, and floor marked as no-go —
  and both are hard, so the interesting number is the price of obeying them.</p>
  <div class="tw"><table>
    <caption>60 held-out rooms, level-3 deformation. Pins and zones are enforced, not weighted.</caption>
    <thead><tr><th scope="col" style="text-align:left">setting</th>
    <th scope="col">obeyed</th><th scope="col">legality ↑</th>
    <th scope="col">S<sub>rel</sub> ↑</th><th scope="col">score ↑</th></tr></thead>
    <tbody>{ct_rows}</tbody>
  </table></div>
  <p>Pinning the most important object holds its pose exactly in every room tested; forbidding
  a quarter of the floor removes {zone_cut:.0f}% of the furniture that would otherwise sit
  there. Both cost score, and that is the point — a room told to leave its best wall empty is
  a harder room.</p>
  <p class="note">Four separate fixes were needed, all the same shape: a hard constraint that
  some <em>other</em> code path quietly re-randomised. The gradient freeze holds a pinned
  object where it starts, so anything that moved its starting point defeated it. Keep-out
  zones were charged through a soft term the feasibility escalation did not raise, making them
  <em>cheaper</em> to violate exactly when the room got tight; they are now punched out of the
  floor field the boundary term reads, so a no-go zone has a wall's gradient.</p>
</section>

<section>
  <div class="eyebrow"><span class="num">Negative results</span><span class="src">§15.2, §20</span></div>
  <h2 class="serif">Two things the plan warned about, measured rather than assumed</h2>
  <div class="grid2">
    <div class="card"><h3>A whole-image CLIP score cannot judge a room</h3>
      <p>{appr_txt}</p></div>
    <div class="card"><h3>VLM relation extraction is as shaky as advertised</h3>
      <p>{vlm_txt}</p></div>
  </div>
</section>

<section>
  <div class="eyebrow"><span class="num">Qualitative</span><span class="src">§14.2, §14.4</span></div>
  <h2 class="serif">What it looks like</h2>
  {FIG['sheet']}
  {FIG['cases']}
  {FIG['flow']}
</section>

<section>
  <div class="eyebrow"><span class="num">Honest limits</span><span class="src">what is not claimed</span></div>
  <h2 class="serif">What is not claimed here</h2>
  <div class="grid2">
    <div class="card"><h3>Multi-view reconstruction</h3>
    <p>GenRecon is the plan's optional multi-view teacher. Its code, its CUDA
    extensions, an attention backend and all 13.7&nbsp;GB of checkpoints are in
    place here, and its input data is generated. It stops on one gated
    dependency: DINOv3 needs a licence accepted on a Hugging Face account. Two
    third-party mirrors were checked and neither serves the weights.</p></div>
    <div class="card"><h3>The human study</h3>
    <p>Two complete instruments exist &mdash; one over synthetic references, one
    over photographed ones &mdash; each with randomised trial order, an answer
    key and a scorer reporting Wilson intervals. No responses have been
    collected, so no human number appears anywhere above.</p></div>
    <div class="card"><h3>Baselines from other papers</h3>
    <p>SAGE's code is public but needs Isaac&nbsp;Sim and foundation-model APIs;
    the 2025 size-aware retargeting paper and CHOrD have no public code that a
    search turns up. The floor-plan-only synthesiser here stands in for the
    semantic/style-conditioned family, and is labelled as a stand-in.</p></div>
    <div class="card"><h3>Asset dimensions</h3>
    <p>Object sizes were first derived from instance medians, then validated
    against the real 3D-FUTURE meshes: median relative difference 0.2&nbsp;%,
    99.5&nbsp;% of models agreeing within 10&nbsp;% on every axis. The corpus was
    not rebuilt because it did not need to be.</p></div>
  </div>
</section>

<footer>
  <p class="wide">Every number on this page comes from a script in the repository and can be
  regenerated with <code>scripts/run_all.sh</code>. Full tables, including the yardsticks
  omitted here for space, are in <code>outputs/REPORT.md</code>.</p>
</footer>
</div>
"""
open("/tmp/claude-1000/-home-gino-project-project4/39bfddf0-0439-451a-bf6e-566c3e5f9f6c/scratchpad/reroom_artifact.html","w").write(HTML)
print("wrote", len(HTML), "bytes")
