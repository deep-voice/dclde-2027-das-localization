# Pitch deck — "Can a fibre hear depth?"

Team pitch for the sound-speed / depth-observability work: from the original sound-speed idea,
through Plan A and the Plan B gate, to what a DCLDE 2027 paper would need.

**Live deck (private until shared):** https://claude.ai/artifact/W3gHMpKk8NizC7Qvonx7vw

## What's here

| Path | What it is |
|---|---|
| `project/deck.json`, `project/slides/*.html` | The published deck's source: one file per slide, speaker notes in each slide's `<aside>` |
| `build_slides.py` | Generates every slide file and `deck.json`. Edit here, not the HTML |
| `../figures/pitch/*.png` | The data figures |
| `../../scripts/make_pitch_figs.py` | Regenerates those figures from the committed result tables and picks |

## Regenerating

```bash
# figures (from repo root; ~2 min, the depth-profile and misfit-field figures rerun the gate's profiler)
PYTHONPATH=. uv run python paper/scripts/make_pitch_figs.py

# slides
python paper/docs/pitch/build_slides.py
```

The slides reference figures by their uploaded asset URLs (`/_blob/<id>`, the `IMG` table at the top
of `build_slides.py`), which resolve only inside the artifact. If you regenerate a figure, re-upload
it to the artifact and update its id in `IMG`, then republish the changed slide files.

## Status of the content

- The exchangeability chart (23 / 918 / 3,044 µs) is the synthetic fixture from
  `experiments/moveout.py::depth_profile`'s docstring, not a committed experiment.
- The geometry-sweep heatmap (`sweep` slide) is a sketch of a proposed figure — not data.
- Illustrations (cover, aperture, rays vs modes, mechanism, noise-floor circles) are schematic.
- Placeholder: DCLDE 2027 submission deadline `[TBD]` on the `paper` slide.
- Open to-dos before pitching are the six items on the `gaps` slide.
