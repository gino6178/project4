# ReRoom bench

A local page for answering the plan's section 14.4 question by eye: does the
retargeted room still read as the reference room's design?

```bash
conda activate reroom
PYOPENGL_PLATFORM=egl PYTHONPATH=. python webapp/server.py --port 8000
# open http://127.0.0.1:8000
```

Everything runs on this machine; nothing leaves it.

**What you can do**
- pick one of 50 held-out 3D-FRONT reference rooms (left column)
- shape a target floor: six presets, or free sliders for width, depth, a corner
  cut and a slanted wall
- toggle what the solver is allowed to do — remove, add, substitute
- see the reference and both results as a photo-like render *and* as a floor
  plan, with the legality and preservation numbers under each
- answer the two questions and have them recorded

**The two questions are deliberately separate.** A result can look like the
reference and still be a bad room; the other way round too. Collapsing them
into "which is better?" is what makes a study like this uninterpretable.

Answers land in `outputs/webapp_votes.jsonl`, one JSON object per judgement,
with the case, the target, both answers and both metric sets. Score them with:

```python
from collections import Counter
import json
rows = [json.loads(l) for l in open("outputs/webapp_votes.jsonl")]
print(Counter(r["q1"] for r in rows))   # preservation
print(Counter(r["q2"] for r in rows))   # suitability
```

**Timing** — a retarget takes 4–7 s, most of it the optimiser; the textured
render adds ~2 s. Untick "render with the real assets" for a faster loop.

**Requirements** — the 3D-FRONT JSONs and the extracted 3D-FUTURE meshes, at
the paths in `--front` / `--future`. Without them the page still works; it just
shows floor plans instead of renders.
