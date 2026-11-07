"""Write the pitch deck's slide files (paper/docs/pitch/project/slides/*.html + project/deck.json).

Run: python paper/docs/pitch/build_slides.py   (see paper/docs/pitch/README.md)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
SL = ROOT / "project" / "slides"
SL.mkdir(parents=True, exist_ok=True)

IMG = {
    "map": "/_blob/edf92ac1802711f21ddc0f22a0ad2024",
    "profile": "/_blob/0b6f690464982e4372990b5f2c493da9",
    "join": "/_blob/9ff2e4063e4ed4e07416cb2d83817dc3",
    "free_c": "/_blob/2b6008f47488fc1a3ec7e8f8a0fadd93",
    "trend": "/_blob/3194dd34ff657ce38bd2844892fb99b3",
    "hyp": "/_blob/bd71c2c89a0a12864541bb3919c454b7",
    "bounce": "/_blob/621500d58029315d05a4eb5cb7664671",
    "sigma": "/_blob/d7a8610c1e6d056412f8f2f2d7041ab5",
    "depth": "/_blob/0d6a89defaff2465040fa2047f318369",
    "fields": "/_blob/3f422222a787103d563e6bcfd46b2828",
    "exchange": "/_blob/b91443d4522c6c2ea09e4d62a4ad37e4",
}

# palette
LIGHT, DARK, ACC_BG = "#F6F4EE", "#0E1F2B", "#0B5F5A"
INK, BODY, MUTE = "#1B2A36", "#4A5864", "#66727C"
TEAL_T, CORAL_T = "#0B7A73", "#B4532A"          # text-safe accents on light
TEAL, CORAL, INDIGO = "#0F9A92", "#D9713F", "#5B6CB0"
D_TEXT, D_BODY, D_ACC = "#F2EDE3", "#B9CCD0", "#5CC8BF"
SERIF = "'Source Serif 4', Georgia, serif"
SANS = "'IBM Plex Sans', Arial, sans-serif"

LIGHT_SEC = (f"background:{LIGHT}; color:{INK}; font-family:{SANS}; padding:128px 128px 160px; "
             "display:flex; flex-direction:column; gap:48px")
DARK_SEC = (f"background:{DARK}; color:{D_TEXT}; font-family:{SANS}; padding:128px 128px 160px; "
            "display:flex; flex-direction:column; gap:48px")

slides = []  # (id, html)


def header(eyebrow, title, dark=False):
    ec = D_ACC if dark else TEAL_T
    tc = D_TEXT if dark else INK
    return (f'<div style="display:flex; flex-direction:column; gap:16px">'
            f'<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{ec}">{eyebrow}</p>'
            f'<h2 style="font-family:{SERIF}; font-size:64px; font-weight:600; line-height:1.1; color:{tc}">{title}</h2>'
            f'</div>')


def footer(text, dark=False, n=None):
    c = "#8FA3A8" if dark else MUTE
    return (f'<p style="position:absolute; left:128px; bottom:64px; width:1664px; font-size:24px; color:{c}">{text}</p>')


def p(text, size=30, color=BODY, extra=""):
    return f'<p style="font-size:{size}px; line-height:1.45; color:{color}; {extra}">{text}</p>'


def notes(t):
    return f"<aside>{t}</aside>"


def add(sid, style, body, note, transition="fade", section_attr=""):
    slides.append((sid, f'<section id="{sid}" data-transition="{transition}" style="{style}">{body}{notes(note)}</section>'))


def img(key, w, h, alt, extra=""):
    return f'<img src="{IMG[key]}" alt="{alt}" style="width:{w}px; height:{h}px; object-fit:contain; {extra}">'


def fig_row(key, w, h, alt, text_html, gap=64, img_first=True):
    left = img(key, w, h, alt)
    right = f'<div style="flex:1; display:flex; flex-direction:column; gap:28px; justify-content:center">{text_html}</div>'
    inner = left + right if img_first else right + left
    return f'<div style="display:flex; flex-direction:row; gap:{gap}px; align-items:center; flex:1">{inner}</div>'


def bignum(num, label, color=TEAL_T, size=96):
    return (f'<div style="display:flex; flex-direction:column; gap:8px">'
            f'<p style="font-family:{SERIF}; font-size:{size}px; font-weight:600; line-height:1.0; color:{color}">{num}</p>'
            f'<p style="font-size:26px; line-height:1.35; color:{BODY}">{label}</p></div>')


def table(rows, widths, size=26, color=INK, head_bg="#E7E3D8", width=1664):
    out = [f'<table style="font-family:{SANS}; font-size:{size}px; color:{color}; width:{width}px">']
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        bg = f' style="background:{head_bg}"' if i == 0 else ""
        cells = []
        for j, c in enumerate(r):
            w = f"width:{widths[j]}%; " if i == 0 else ""
            cells.append(f'<{tag} style="{w}text-align:left">{c}</{tag}>')
        out.append(f"<tr{bg}>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def divider(sid, num, title, sub, note):
    body = (f'<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{D_ACC}">{num}</p>'
            f'<h1 style="font-family:{SERIF}; font-size:112px; font-weight:600; line-height:1.05; color:{D_TEXT}; width:1400px">{title}</h1>'
            f'<p style="font-size:36px; line-height:1.4; color:{D_BODY}; width:1300px">{sub}</p>')
    style = (f"background:{DARK}; color:{D_TEXT}; font-family:{SANS}; padding:128px; display:flex; "
             "flex-direction:column; justify-content:center; gap:32px")
    add(sid, style, body, note)


# ============================================================ 1 cover
cover_svg = f'''<svg aria-label="Illustration: a whale call spreading through the water toward a fibre cable lying on a seafloor that slopes from shelf to abyss" width="1920" height="520" viewBox="0 0 1920 520" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#123447"/><stop offset="1" stop-color="#0B1A24"/></linearGradient></defs>
<rect x="0" y="40" width="1920" height="480" fill="url(#w)"/>
<line x1="0" y1="40" x2="1920" y2="40" stroke="#5CC8BF" stroke-width="2" opacity="0.6"/>
<g fill="none" stroke="#5CC8BF">
<circle cx="1180" cy="92" r="110" stroke-width="3" opacity="0.55"/>
<circle cx="1180" cy="92" r="230" stroke-width="3" opacity="0.4"/>
<circle cx="1180" cy="92" r="360" stroke-width="3" opacity="0.28"/>
<circle cx="1180" cy="92" r="500" stroke-width="3" opacity="0.18"/>
<circle cx="1180" cy="92" r="650" stroke-width="3" opacity="0.1"/>
</g>
<path d="M1920 150 L1500 160 C1380 165 1300 190 1200 240 C1000 340 760 430 420 470 L0 490 L0 520 L1920 520 Z" fill="#0A141B"/>
<path d="M1920 146 L1500 156 C1380 161 1300 186 1200 236 C1000 336 760 426 420 466 L0 486" fill="none" stroke="#F2E6C9" stroke-width="4" stroke-dasharray="2 10" stroke-linecap="round"/>
<g transform="translate(1180 92)">
<path d="M-70 0 C-40 -26 40 -26 70 -4 C80 2 80 8 70 10 C40 22 -40 22 -70 0 Z" fill="#D9713F"/>
<path d="M-68 0 L-104 -22 L-96 0 L-104 20 Z" fill="#D9713F"/>
</g>
</svg>'''
add("cover",
    f"background:{DARK}; color:{D_TEXT}; font-family:{SANS}; padding:128px; display:flex; flex-direction:column; gap:28px",
    cover_svg.replace("<svg ", '<svg style="position:absolute; left:0px; top:560px; width:1920px; height:520px" ', 1)
    + f'<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{D_ACC}">Deep Voice · DCLDE 2027 · DAS fin-whale localization</p>'
    + f'<h1 style="font-family:{SERIF}; font-size:120px; font-weight:600; line-height:1.05; color:{D_TEXT}">Can a fibre hear depth?</h1>'
    + f'<p style="font-size:36px; line-height:1.4; color:{D_BODY}; width:1400px">From a sound-speed idea to a clean negative result — and a paper about when seafloor cables can and cannot locate a whale in 3-D.</p>',
    "Pitch to the team. The arc: I had an idea to bring real ocean data into DAS localization. We tested it properly (Plan A), then tested the depth question it was really about (Plan B). The answer is no on this dataset, but the reason is geometric and general, which is what makes it publishable. Illustration on this slide is schematic, not to scale.")

# ============================================================ 2 setup (map)
add("setup", DARK_SEC,
    header("The data", "Two fibres off Oregon, running shelf to abyss", dark=True)
    + '<div style="display:flex; flex-direction:row; gap:56px; align-items:center; flex:1">'
    + f'<div style="width:420px; display:flex; flex-direction:column; gap:36px">'
    + "".join(f'<div style="display:flex; flex-direction:column; gap:4px"><p style="font-family:{SERIF}; font-size:64px; font-weight:600; line-height:1.0; color:{D_ACC}">{n}</p><p style="font-size:26px; line-height:1.3; color:{D_BODY}">{l}</p></div>'
              for n, l in [("628", "annotated fin-whale calls, Nov 2021"), ("11,117", "arrival-time picks"), ("84–1,555 m", "depth range of the picked channels")])
    + '</div>'
    + f'<img src="{IMG["map"]}" alt="Bathymetric map of the Oregon margin with the North and South DAS cables and one circle per solved call position" style="width:1188px; height:526px; object-fit:contain">'
    + '</div>'
    + footer("Map: GMRT bathymetry, cables (cream), and each call's solved position under the standard model (orange, n = 498).", dark=True),
    "This is the DCLDE 2027 DAS challenge dataset. Two telecom fibres used as thousands of strain sensors. Depth shading goes from shallow shelf (light, right, near the coast) to deep water (dark, left). The key physical fact for everything that follows: the channels on these cables sit anywhere from about 84 m to 1,555 m deep. Circles are positions from our ConstantC solver at the standard assumptions (c = 1480 m/s, source at -30 m), from the free-c ablation table's fixed-c columns.")

# ============================================================ 3 standard practice
add("practice", LIGHT_SEC,
    header("Standard practice", "Everyone assumes one sound speed and a fixed depth")
    + table([["Study", "Sound speed used", "Source depth"],
             ["Goestchel, Wilcock &amp; Abadi 2026 (this dataset)", "1480 m/s, constant", "fixed at −30 m"],
             ["Horne, Stork &amp; Stanek 2025 (same Oregon data)", "1480 m/s", "fixed at 10 m"],
             ["Bouffaut et al. 2022 (Svalbard)", "1500 m/s", "fixed at 30 m"],
             ["Rørstadbotnen et al. 2023 (Svalbard, two cables)", "1440 m/s, air-gun calibrated", "fixed at 20 m"]],
            [48, 30, 22], size=30)
    + p("Nobody estimates depth, so “3-D” DAS whale localization doesn’t exist yet. The DCLDE 2027 challenge asks for it anyway.", 32, INK),
    "Every published DAS whale localization study uses a single constant sound speed and pins the source depth. We verified in the baseline authors' own released output that every row has z = -30.0 m exactly (their single-cable HDF5 output uses -40 m). Task L-2 of the challenge asks for 3-D localization, which nobody has done.")

# ============================================================ 4 the idea
steps = [("1", "Pull the ocean", "HYCOM reanalysis T/S, OOI cabled-array CTD casts, GMRT bathymetry for Nov 2021."),
         ("2", "Compute sound speed", "TEOS-10 c(z) at the cable site — the water isn’t 1480 m/s everywhere."),
         ("3", "Invert with it", "Replace the constant in the arrival-time inversion; see what moves.")]
step_html = "".join(
    f'<div style="display:flex; flex-direction:row; gap:28px; align-items:start">'
    f'<p style="font-family:{SERIF}; font-size:56px; font-weight:600; line-height:1.0; color:{TEAL_T}; width:48px">{n}</p>'
    f'<div style="flex:1; display:flex; flex-direction:column; gap:6px"><h3 style="font-size:36px; font-weight:600; color:{INK}">{t}</h3>'
    f'<p style="font-size:28px; line-height:1.4; color:{BODY}">{d}</p></div></div>' for n, t, d in steps)
add("idea", LIGHT_SEC,
    header("The idea", "Use the real ocean, not a constant")
    + '<div style="display:flex; flex-direction:row; gap:72px; align-items:center; flex:1">'
    + img("profile", 520, 618, "Sound-speed profile at the South cable's deep end, from 1497 m/s at the surface down to a 1477 m/s minimum near 400-600 m, with the cable channel depth band and the CTD-validated band shaded")
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:40px">{step_html}</div></div>',
    "My starting idea. Real sound speed at this site varies from about 1497 m/s at the surface to a 1477 m/s minimum near 400-600 m. The plot is the HYCOM profile at the South cable's deepest channel. Grey band: depth range of the cable channels. Teal band: the only depths where we could check HYCOM against in-situ CTD (219-573 m, median error 0.47 m/s, but 87 km away at a shallower site). Getting that profile honestly took about a day: OOI's deep profiler at Slope Base had no data for Nov 2021, two instruments were parked, and HYCOM's CSV endpoint returns packed integers that decode to -6483 deg C if you're naive.")

# ============================================================ 5 why depth (illustration)
aperture_svg = '''<svg aria-label="Side-view illustration: a shallow whale and two cable channels, one on the shelf and one in deep water; the two rays pass through different sound-speed layers" width="1000" height="560" viewBox="0 0 1000 560" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="c" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE7E4"/><stop offset="0.35" stop-color="#9FCFCB"/><stop offset="0.6" stop-color="#7DB9BC"/><stop offset="1" stop-color="#8FC3C6"/></linearGradient></defs>
<rect x="0" y="40" width="1000" height="520" fill="url(#c)"/>
<line x1="0" y1="40" x2="1000" y2="40" stroke="#1B2A36" stroke-width="2"/>
<path d="M1000 120 L880 128 C780 140 700 190 600 260 C460 360 300 440 120 480 L0 492 L0 560 L1000 560 Z" fill="#D8D3C4"/>
<path d="M1000 118 L880 126 C780 138 700 188 600 258 C460 358 300 438 120 478 L0 490" fill="none" stroke="#1B2A36" stroke-width="4" stroke-dasharray="3 9" stroke-linecap="round"/>
<line x1="560" y1="78" x2="870" y2="124" stroke="#0F9A92" stroke-width="5"/>
<line x1="560" y1="78" x2="150" y2="472" stroke="#D9713F" stroke-width="5"/>
<circle cx="870" cy="124" r="12" fill="#0F9A92" stroke="#F6F4EE" stroke-width="3"/>
<circle cx="150" cy="472" r="12" fill="#D9713F" stroke="#F6F4EE" stroke-width="3"/>
<g transform="translate(560 78)"><path d="M-34 0 C-20 -13 20 -13 34 -2 C40 1 40 4 34 5 C20 11 -20 11 -34 0 Z" fill="#1B2A36"/><path d="M-33 0 L-52 -11 L-48 0 L-52 10 Z" fill="#1B2A36"/></g>
</svg>'''
add("aperture", LIGHT_SEC,
    header("The hypothesis", "Why sound speed might unlock depth")
    + '<div style="display:flex; flex-direction:row; gap:64px; align-items:center; flex:1">'
    + '<div style="position:relative; width:1000px; height:560px">' + aperture_svg
    + f'<p style="position:absolute; left:640px; top:152px; width:340px; font-size:24px; color:{TEAL_T}; font-weight:600">shelf channel ~100 m</p>'
    + f'<p style="position:absolute; left:180px; top:500px; width:360px; font-size:24px; color:{CORAL_T}; font-weight:600">deep channel ~1,500 m</p>'
    + f'<p style="position:absolute; left:600px; top:0px; width:200px; font-size:24px; color:{INK}">whale</p>'
    + '</div>'
    + '<div style="flex:1; display:flex; flex-direction:column; gap:28px">'
    + f'<p style="font-family:{SERIF}; font-size:40px; color:{INK}">∂t / ∂zₛ = (zₛ − z_c) / (c · R)</p>'
    + p("Depth only shows up because channel depth z_c changes along the cable. These cables span 1,500 m of it.", 28)
    + p("But rays to shallow and deep channels cross different water — so a constant c would <b>bias</b> depth exactly where the depth signal lives.", 28)
    + '</div></div>'
    + footer("Illustration — not to scale. The equation is our inference from the forward model; no paper had tested it."),
    "The real motivation. With two cables at similar depth, depth is unobservable (Rorstadbotnen moved depth 15-30 m and horizontal position moved under 1 m). But these cables go from ~100 m to ~1,550 m, which gives vertical aperture. And the ray to a deep channel samples the whole profile, including the sound-channel minimum, while the ray to a shelf channel never leaves the upper layer. So the sound-speed profile would stop being a small correction and become a prerequisite for depth. That was the hypothesis going in.")

# ============================================================ 6 divider Plan A
divider("div-a", "Plan A", "Does real sound speed change anything?",
        "Reproduce the baseline, characterise the propagation regime, run two sound-speed ablations — with the criteria fixed before running.",
        "Plan A was a decision gate: stop at the first negative. It was explicitly allowed to conclude the programme was aimed at the wrong term.")

# ============================================================ 7 scorecard
add("scorecard", LIGHT_SEC,
    header("Plan A in one table", "Five questions, five answers")
    + table([["Question", "Answer", "Key number"],
             ["Can we reproduce the published baseline per call?", "No — the release has no call ID", "0 / 176 rows join at 90 s"],
             ["Does fitting c as a free parameter help?", "It soaks up errors, not sound speed", "4.6% of 498 fits plausible"],
             ["Does a depth-dependent c move positions?", "Yes — below the noise floor", "9–9.5 m vs 142 m (n = 459)"],
             ["Is the residual range trend a depth artifact?", "No", "76–82 ms at every depth"],
             ["Is there really a ray shadow zone?", "No — trapped modes exist", "≥ 1 mode in 5 / 5 geometries"]],
            [42, 32, 26], size=28),
    "Summary of Plan A. Each row gets its own slide next. The headline: water-column sound speed is demonstrably not the lever for horizontal position on this data. The depth question was left for Plan B.")

# ============================================================ 8 baseline join
add("join", LIGHT_SEC,
    header("Plan A · 1", "The published baseline can’t be checked call by call")
    + fig_row("join", 900, 560, "Stacked bars of the 176 baseline rows by timestamp tolerance: all ambiguous at 90, 30 and 10 seconds; at most 33 matched at 1 second, with 109 unmatched",
              bignum("0 / 176", "baseline rows join uniquely to a call at the default 90 s tolerance")
              + p("Calls arrive every ~4 s; the only join key is a timestamp. Tighten it and rows fall out instead.", 28)
              + p("Worth raising with the organizers: a localization set with no call ID can’t be verified by anyone.", 28, INK)),
    "Goestchel et al.'s released CSV (176 rows, batch 1) has no call identifier. Every row has at least 10 candidate calls within 90 s (median 30). Best case is 33/176 at 1 s, bought by pushing 109 outside tolerance. Our reproduction run used a timing-isolation rule: 21 uncontested, 17 solved, median distance 247 m, but that is a biased sample and NOT a reproduction accuracy. Also the baseline's own association uses a median 4,684 picks per call vs our 14, so the two estimators aren't solving the same input.")

# ============================================================ 9 free c
add("freec", LIGHT_SEC,
    header("Plan A · 2", "A free sound speed soaks up errors, not physics")
    + fig_row("free_c", 1000, 560, "Histogram of fitted sound speed for 498 calls: large spikes at both search-bracket edges (1430 and 1530 m/s) and very few inside the physically plausible band 1477.5-1497.5 m/s",
              bignum("4.6%", "of 498 fitted values fall in the physically plausible band")
              + bignum("41.8%", "pin to the edge of the search range", CORAL_T)
              + p("Real sound speed doesn’t swing ±50 m/s between calls seconds apart.", 28)),
    "Fitting c per call over [1430, 1530] m/s. Median fitted c is 1450 m/s. Plausible band from the profile's own range: 1477.5-1497.5. 23/498 plausible; 208/498 pinned (149 low edge, 59 high). Calls whose fitted c is plausible move 4x less than calls whose c is unphysical (58.7 m vs 250.1 m median). So the cheap version of the idea - just let c float - does not work; it's a nuisance sink for model and association error.")

# ============================================================ 10 below noise (illustration)
circ_svg = '''<svg aria-label="Two concentric circles drawn to scale: a 142 metre pick-noise radius and a 9 metre position shift" width="600" height="600" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
<circle cx="300" cy="300" r="284" fill="#D9713F" fill-opacity="0.14" stroke="#D9713F" stroke-width="3" stroke-dasharray="10 8"/>
<circle cx="300" cy="300" r="18" fill="#0F9A92"/>
</svg>'''
add("noisefloor", LIGHT_SEC,
    header("Plan A · 3", "The real correction is too small to see here")
    + '<div style="display:flex; flex-direction:row; gap:96px; align-items:center; flex:1">'
    + '<div style="position:relative; width:600px; height:600px">' + circ_svg + '</div>'
    + '<div style="flex:1; display:flex; flex-direction:column; gap:40px">'
    + bignum("9–9.5 m", "median horizontal shift from a depth-dependent sound speed (n = 459 stable calls of 489)", TEAL_T)
    + bignum("142 m", "this dataset’s pick-noise floor (~96 ms per-call residual × 1480 m/s)", CORAL_T)
    + p("6.3–6.7% of the floor: not zero, but unresolvable at this SNR.", 30, INK)
    + '</div></div>'
    + footer("Circles drawn to scale. Profile: one reanalysis product (HYCOM), never checked in situ at its own location."),
    "PathAverage and RayTrace effective-c models vs ConstantC(1480). n = 489 calls both sides solved; 459 basin-stable, 30 (6.1%) flip basins and move a median 1,273 m — never merged into the 9 m figure. Caveat: RayTrace falls back to PathAverage for 99.4% of search-grid queries, so the two rows aren't independent. The 142 m floor is an upper bound on pick noise since residuals include model error. Correct statement: we cannot distinguish the effect from zero on this data, not that it is zero.")

# ============================================================ 11 trend
add("trend", LIGHT_SEC,
    header("Plan A · 4", "The range trend isn’t a depth artifact either")
    + fig_row("trend", 900, 560, "Far-minus-near residual trend stays between 76 and 82 milliseconds as assumed source depth goes from 5 to 200 metres, with 95 percent error bars",
              bignum("76–82 ms", "far − near residual at every assumed depth from 5 to 200 m (≈ 500 calls each)")
              + p("A 40× change in assumed depth barely moves it. Something range-structured is wrong in the model — but it isn’t source depth.", 28)
              + p("Still unidentified: path, mis-association, bathymetry or clock error.", 28, INK)),
    "Mean pick residual rises with source-receiver range, fastest at highest SNR (so not just weak picks). We swept the assumed source depth from -5 to -200 m with ConstantC(1480). The trend (primary statistic, |r| < 1 s cut) stays 75.8-81.8 ms; error bars are 1.96 x naive SE (5.1-5.2 ms). 497-502 calls solved per depth. This falsifies one explanation by elimination but doesn't identify the cause. Consequence: Plan B couldn't use 'the trend is a depth artifact' as its premise.")

# ============================================================ 12 shadow vs modes (illustration)
ray_svg = '''<svg aria-label="Ray picture: rays from a shallow source bend downward into the seabed, leaving a shaded shadow zone near a distant shallow receiver" width="760" height="420" viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="20" width="760" height="400" fill="#DCEEEC"/>
<line x1="0" y1="20" x2="760" y2="20" stroke="#1B2A36" stroke-width="2"/>
<path d="M300 22 L760 22 L760 150 C600 150 450 110 300 22 Z" fill="#D9713F" fill-opacity="0.22"/>
<path d="M0 360 C200 370 500 380 760 390 L760 420 L0 420 Z" fill="#D8D3C4"/>
<g fill="none" stroke="#5B6CB0" stroke-width="3">
<path d="M40 50 C200 60 330 150 420 372"/><path d="M40 50 C180 70 260 170 300 368"/><path d="M40 50 C140 90 190 190 200 366"/><path d="M40 50 C260 55 430 140 560 381"/>
</g>
<circle cx="40" cy="50" r="11" fill="#1B2A36"/>
<circle cx="690" cy="92" r="11" fill="#D9713F" stroke="#F6F4EE" stroke-width="3"/>
</svg>'''
mode_svg = '''<svg aria-label="Mode picture: three standing-wave mode shapes spanning the water column carry energy along the waveguide" width="760" height="420" viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="20" width="760" height="400" fill="#DCEEEC"/>
<line x1="0" y1="20" x2="760" y2="20" stroke="#1B2A36" stroke-width="2"/>
<path d="M0 360 C200 370 500 380 760 390 L760 420 L0 420 Z" fill="#D8D3C4"/>
<g fill="none" stroke-width="4">
<path d="M150 22 C230 120 230 260 150 380" stroke="#0F9A92"/>
<path d="M360 22 C440 90 440 150 360 200 C280 250 280 330 360 380" stroke="#5B6CB0"/>
<path d="M570 22 C630 60 630 100 570 140 C510 180 510 220 570 260 C630 300 630 340 570 380" stroke="#D9713F"/>
</g>
<g stroke="#8A949C" stroke-width="2"><line x1="150" y1="22" x2="150" y2="380"/><line x1="360" y1="22" x2="360" y2="380"/><line x1="570" y1="22" x2="570" y2="380"/></g>
</svg>'''
add("modes", LIGHT_SEC,
    header("Plan A · 5", "Ray theory says “shadow”; modes say “weak but open”")
    + '<div style="display:flex; flex-direction:row; gap:48px; flex:1">'
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:20px">{ray_svg}<h3 style="font-size:32px; font-weight:600; color:{INK}">Rays: no direct path</h3>'
    + p("A downward-refracting profile bends every ray into the seabed — the distant receiver sits in shadow.", 26) + '</div>'
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:20px">{mode_svg}<h3 style="font-size:32px; font-weight:600; color:{INK}">Modes: at least one survives</h3>'
    + p("At 20 Hz the guide is only 1–21 wavelengths deep. ≥ 1 trapped mode in all 5 “shadowed” geometries, every seabed.", 26) + '</div>'
    + '</div>'
    + footer("Illustration — schematic. Mode solver validated 224/224 against the analytic Pekeris waveguide."),
    "Ray tracing called all five test geometries (5-40 km, receivers 100-1575 m) shadowed. A normal-mode solver found 1-2 modes at the shallowest up to 13-27 at the deepest, for every seabed in a 1550-1900 m/s sweep. So the argument to abandon the straight-line model was a ray artifact. Follow-up transmission-loss work: modes exist but a 20 Hz source at 30 m barely excites them (excitation_rms never above 0.075), and shallow water strips about half of them by 20 km.")

# ============================================================ 13 path, two ways
add("path", LIGHT_SEC,
    header("Plan A follow-up", "The first arrival looks direct, from two independent angles")
    + '<div style="display:flex; flex-direction:row; gap:48px; flex:1">'
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:16px">'
    + img("hyp", 720, 474, "Delta log-likelihood of the trapped-mode hypothesis minus direct path, for five seabed sound speeds: all between minus 972 and minus 1228, intervals far from zero")
    + f'<h3 style="font-size:32px; font-weight:600; color:{INK}">Amplitude: direct fits better at every seabed</h3>'
    + p("Δ log-lik −972 to −1,228, n = 5,349 picks / 380 calls.", 26) + '</div>'
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:16px">'
    + img("bounce", 720, 448, "Histogram of per-call delta chi-squared, bounce minus direct: 235 of 266 calls favour the direct path, with a small bounce-favouring tail")
    + f'<h3 style="font-size:32px; font-weight:600; color:{INK}">Timing: 235 of 266 calls favour direct</h3>'
    + p("136 of those reach 1σ; a real 7-call bounce tail at 2σ.", 26) + '</div>'
    + '</div>'
    + footer("Amplitude caveat: the SNR scale fails its own geometric calibrator (0.16 vs 2). Timing has no dB scale, which is why it matters."),
    "Transmission-loss study: pre-registered comparison of two fixed shapes for pick SNR vs range and depth. H-geometric (direct path) beats H-modal (adiabatic trapped modes) by ~1,000 log-likelihood units at all five seabeds; the modal hypothesis also predicts non-detection for 3,051-3,311 of 8,660 picks that were detected. Two big caveats: it's comparative (neither is shown to fit in absolute terms), and the axial-projection coefficient comes out at 0.15-0.17 against a physics benchmark of 2 - a compression that would bias toward the direct path. That's why the timing check matters: direct vs flat-bottom bounce on the same picks; dotted lines on the right plot are the 2-sigma thresholds. Also: received level carries no depth information (depth coefficient -0.78 dB/km, 95% CI [-2.10, +0.73]).")

# ============================================================ 14 divider Plan B
divider("div-b", "Plan B", "Is source depth recoverable from moveout?",
        "Amplitude was already closed. The last route to depth is arrival-time curvature across the cable. We set a pass/fail gate before running it.",
        "Plan B's gate: an identifiability precondition before building any depth-estimation machinery. Failing it was pre-declared as a complete, publishable outcome.")

# ============================================================ 15 the gate
crit = (f'<div style="width:720px; display:flex; flex-direction:column; gap:20px; background:#EAE6DB; padding:44px; border-radius:16px">'
        f'<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{TEAL_T}">Pre-registered criterion</p>'
        f'<p style="font-size:30px; line-height:1.45; color:{INK}">Among calls with ≥ 8 picks, ≥ 200 m of receiver-depth span, basin-stable: pass only if <b>≥ 10% and ≥ 50 calls</b> reach a depth uncertainty σz ≤ 100 m.</p>'
        f'<p style="font-size:26px; line-height:1.4; color:{BODY}">100 m, because the null already assumes −30 m and fin whales call shallow.</p></div>')
res = ('<div style="flex:1; display:flex; flex-direction:column; gap:44px">'
       + bignum("17 / 271", "calls reach σz ≤ 100 m — 6.3%, fails both the 10% and the 50-call bar", CORAL_T, 120)
       + bignum("2 / 271", "survive an extra consistency check (walking the depth profile both ways)", CORAL_T, 72)
       + '</div>')
add("gate", LIGHT_SEC,
    header("Plan B · the gate", "The gate fails")
    + f'<div style="display:flex; flex-direction:row; gap:80px; align-items:center; flex:1">{crit}{res}</div>',
    "Stratum: 271 of 628 calls (156 two-cable, 115 single-cable). Statistic: marginal 1-sigma on source depth from profiling x and y out of the misfit at each depth, t0 analytic. A linearised estimate before the run suggested 15-18% would pass, so the bar was genuinely in play. The walk-consistency condition was added after the pre-registration, so we lead with 17, not 2. Even stacking every charitable assumption at once gives ~33 calls, still under 50.")

# ============================================================ 16 sigma breakdown
bd = table([["Outcome (of 271)", "Calls"],
            ["No finite width — grid floor at −600 m", "114"],
            ["No finite width — misfit never rises enough", "98"],
            ["No finite width — minimum at the surface", "17"],
            ["Finite, but σz &gt; 100 m", "25"],
            ["σz ≤ 100 m, artifact", "15"],
            ["σz ≤ 100 m, survives", "2"]], [78, 22], size=26)
add("sigma", LIGHT_SEC,
    header("Plan B · the gate", "Most calls don’t bound depth at all")
    + '<div style="display:flex; flex-direction:row; gap:56px; align-items:center; flex:1">'
    + img("sigma", 860, 507, "Histogram of finite depth uncertainty for 42 calls: median 126 m, most above the 100 m target line")
    + f'<div style="flex:1; display:flex; flex-direction:column; gap:24px">{bd.replace("width:1664px", "width:748px")}'
    + p("Only 42 of 271 get a finite σz; their median is 126 m.", 26, INK) + '</div></div>',
    "Plot: the 42 calls with a finite marginal width; dashed line is the 100 m target. Table: all 271 stratum calls. The 114 grid-floor calls are a real limitation of this run - a probe of 20 of them on a -2000 m grid found interior minima for 17 but sigma_z <= 100 m for none (not table-backed yet; needs to be committed before a paper). The robust residual sigma used for chi-squared is if anything slightly narrower than the RMS, so it's not an over-wide noise estimate.")

# ============================================================ 17 depth profiles
add("profiles", LIGHT_SEC,
    header("Plan B · what the misfit looks like", "Three calls, three ways depth fails or holds")
    + fig_row("depth", 950, 600, "Delta chi-squared versus source depth for three calls: a two-cable call bounded around 200 m, a two-cable call that stays flat below 1 across 0-600 m, and a single-cable call whose downward and upward walks trace different curves",
              f'<div style="display:flex; flex-direction:column; gap:6px"><h3 style="font-size:30px; font-weight:600; color:{TEAL_T}">Bounded</h3>' + p("One of the 2 survivors: σz 78 m, two cables.", 26) + '</div>'
              + f'<div style="display:flex; flex-direction:column; gap:6px"><h3 style="font-size:30px; font-weight:600; color:{INDIGO}">Flat</h3>' + p("Never crosses Δχ² = 1: any depth 0–600 m fits.", 26) + '</div>'
              + f'<div style="display:flex; flex-direction:column; gap:6px"><h3 style="font-size:30px; font-weight:600; color:{CORAL_T}">Path-dependent</h3>' + p("One cable: walking down vs. up gives two curves. A narrow σz here is a local-minimum artifact.", 26) + '</div>'
              + footer("Recomputed from the committed picks with the gate’s own profiling code. Dotted line: Δχ² = 1 (1σ).")),
    "Real data, recomputed with depth_profile() from the gate module. Calls: 2021-11-04_02_05_02/7 (teal, survivor), 2021-11-03_22_10_42/1 (indigo, not_crossed), 2021-11-04_08_02_32/7 (coral, forward walk jumps at ~120 m; the reverse walk tracks a different basin down to 600 m). Four independent reasons the 15 artifacts aren't real: direction-dependent curves (discrepancy 1e-3 to 1e-2 s vs 1e-13 for consistent calls), invariant to a 1000x tighter optimiser, all single-cable, and anomalously small residual sigma (6-27 ms vs 98 ms median).")

# ============================================================ 18 fields (visualizer-inspired)
add("fields", DARK_SEC,
    header("Plan B · on the map", "Push the whale 420 m deeper and the fit just slides", dark=True)
    + '<div style="position:relative; width:1664px; height:600px">'
    + f'<img src="{IMG["fields"]}" alt="Two misfit fields for one single-cable call on bathymetry: at 30 m depth the best fit sits on the cable; at 450 m it slides about 750 m off the cable and fits as well" style="position:absolute; left:0px; top:0px; width:1664px; height:600px; object-fit:contain">'
    + f'<p style="position:absolute; left:195px; top:16px; width:600px; font-size:28px; font-weight:600; color:{D_ACC}">source at 30 m · RMS misfit 18.4 ms</p>'
    + f'<p style="position:absolute; left:852px; top:16px; width:600px; font-size:28px; font-weight:600; color:#F0A57C">source at 450 m · RMS misfit 14.6 ms</p>'
    + '</div>'
    + footer("One single-cable call (14 picks). Colour: misfit within 3× of the best; contours at 1.1, 1.5 and 2× the best fit. Style from our visualizer.", dark=True),
    "Visualizer-style view of the real misfit field for call 2021-11-04_08_02_32/7 (South cable, 14 picks), computed with forward.misfit_field (L2) on a 7 km box. At an assumed depth of 30 m the minimum sits on the cable; at 450 m the minimum moves about 750 m off-axis and the fit is no worse - actually slightly better. That is depth trading off against horizontal position. Note the second lobe on the other side at 450 m: the mirror ambiguity appears as the source goes deeper.")

# ============================================================ 19 mechanism (illustration)
mech_svg = '''<svg aria-label="Side view: a cable on a monotonic slope; source A shallow and source B deeper and shifted along the slope give almost the same distances to every channel" width="900" height="520" viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="20" width="900" height="500" fill="#DCEEEC"/>
<line x1="0" y1="20" x2="900" y2="20" stroke="#1B2A36" stroke-width="2"/>
<path d="M900 90 L0 470 L0 520 L900 520 Z" fill="#D8D3C4"/>
<g fill="#1B2A36"><circle cx="820" cy="124" r="8"/><circle cx="680" cy="183" r="8"/><circle cx="540" cy="242" r="8"/><circle cx="400" cy="301" r="8"/><circle cx="260" cy="360" r="8"/><circle cx="120" cy="419" r="8"/></g>
<g stroke-width="2" opacity="0.7"><line x1="560" y1="70" x2="820" y2="124" stroke="#0F9A92"/><line x1="560" y1="70" x2="540" y2="242" stroke="#0F9A92"/><line x1="560" y1="70" x2="120" y2="419" stroke="#0F9A92"/>
<line x1="470" y1="150" x2="820" y2="124" stroke="#D9713F" stroke-dasharray="8 6"/><line x1="470" y1="150" x2="540" y2="242" stroke="#D9713F" stroke-dasharray="8 6"/><line x1="470" y1="150" x2="120" y2="419" stroke="#D9713F" stroke-dasharray="8 6"/></g>
<path d="M552 80 L482 138" stroke="#1B2A36" stroke-width="3" fill="none"/><path d="M474 146 L480 126 L494 140 Z" fill="#1B2A36"/>
<circle cx="560" cy="70" r="16" fill="#0F9A92" stroke="#F6F4EE" stroke-width="3"/>
<circle cx="470" cy="150" r="16" fill="#D9713F" stroke="#F6F4EE" stroke-width="3"/>
</svg>'''
add("mechanism", LIGHT_SEC,
    header("Why it fails", "On a steady slope, deeper looks like further along")
    + '<div style="display:flex; flex-direction:row; gap:64px; align-items:center; flex:1">'
    + '<div style="position:relative; width:900px; height:520px">' + mech_svg
    + f'<p style="position:absolute; left:590px; top:40px; width:260px; font-size:24px; font-weight:600; color:{TEAL_T}">A: shallow</p>'
    + f'<p style="position:absolute; left:300px; top:140px; width:160px; font-size:24px; font-weight:600; color:{CORAL_T}">B: deeper, shifted</p>'
    + '</div>'
    + '<div style="flex:1; display:flex; flex-direction:column; gap:28px">'
    + p("When channel depth changes steadily along the cable, a deeper source and a source slid along the cable push every arrival time the same way.", 30, INK)
    + p("The solver re-fits x and y and absorbs almost the whole depth change. What’s left is smaller than the pick noise.", 28)
    + p("More or better picks don’t fix this. Geometry does.", 30, TEAL_T, "font-weight:600")
    + '</div></div>'
    + footer("Illustration — schematic side view, not to scale."),
    "The mechanism behind the gate failure, and the transferable result. Depth and along-cable source position are nearly the same parameter when receiver depth is monotonic along the cable, because both perturbations move every receiver's travel time in the same direction. It explains why 16 of the 17 pre-declared 'resolutions' came from single-cable calls. This is a property of the sensitivity structure, not of the velocity model, so it holds regardless of the propagation debate.")

# ============================================================ 20 exchange chart
add("exchange", LIGHT_SEC,
    header("Why it fails", "Two cables at different bearings break the tie")
    + fig_row("exchange", 950, 560, "Log-scale bar chart of residual left after 400 m of depth error and refitting: 23 microseconds on a monotonic single cable, 918 on a V-shaped cable, 3044 with two cables at different bearings",
              bignum("130×", "more depth signal survives with two cables at different bearings than on one monotonic arc", TEAL_T)
              + p("23 µs is below the optimiser’s own positional slop (~34 µs). On one monotonic arc, the signal and the numerical noise are the same size.", 28)
              + p("This is what turns the result into cable-layout advice.", 28, INK)),
    "From the synthetic fixture in the depth_profile docstring: 400 m of source-depth error, then refit x and y; RMS residual left over. 23 us for a single monotonic arc, 918 us for a V-shaped single arc, 3044 us for two arcs at clearly different bearings. 3044/23 = ~132x. IMPORTANT for the paper: this is currently a three-point docstring fixture, not a committed experiment. Turning it into a proper sweep is the #1 gap.")

# ============================================================ 20b depth signal vs noise
add("signal", LIGHT_SEC,
    header("Why it fails", "More picks don’t buy depth")
    + '<div style="display:flex; flex-direction:row; gap:96px; align-items:center; flex:1">'
    + '<div style="display:flex; flex-direction:column; gap:48px; width:700px">'
    + bignum("0.79", "median Δχ² for moving the source 100 m deeper, with x and y refit (271 calls). Below 1 for most calls.", TEAL_T)
    + bignum("13 ms", "depth signal left after the refit, against 98 ms of noise on each pick", CORAL_T)
    + '</div>'
    + '<div style="flex:1; display:flex; flex-direction:column; gap:32px">'
    + p("Four unknowns need four picks. A typical call has 14, and 627 of 628 have at least 4. The count isn’t the limit.", 28, INK)
    + table([["Picks per call", "Calls", "Median Δχ²", "Reach 2σ"],
             ["8–15", "65", "1.38", "14"],
             ["16–25", "75", "1.00", "13"],
             ["26–40", "93", "0.47", "4"],
             ["41+", "38", "0.69", "1"]],
            [34, 20, 26, 20], size=26, width=860)
    + p("Calls with more picks are no better at depth.", 28, TEAL_T, "font-weight:600")
    + '</div></div>'
    + footer("Denominator: the 271-call gate stratum. Δχ² = n·(m₁² − m₀²)/σ², absolute value, source −30 m → −130 m."),
    "Answers the team's question: with two cables there are far more than four picks, so why can't we triangulate depth? Because the picks are not the bottleneck. A call has a median of 14 picks (mean 17.7, range 3-61; 627 of 628 have at least 4). Moving the source 100 m deeper and re-fitting x, y and t0 changes the misfit by a median |dchi2| of 0.787 over the 271 gate-stratum calls: 122 calls reach 1 sigma, 32 reach 2 sigma, 17 reach 3 sigma. In rms terms the leftover depth signal is a median 12.7 ms against a median 97.8 ms of per-pick noise. By picks per call the median |dchi2| does not rise (1.38, 1.00, 0.47, 0.69). Two-cable calls are NOT better on this measure (median 0.47 against 1.60 for one cable, 5 against 27 calls at 2 sigma), but they are also farther from the cables, so that comparison is confounded. Caveats: single-direction profiles, so the walk-consistency check of the gate is not applied and the 2 sigma tail is partly the direction-dependent artifacts the gate found. Read the median, not the tail. Annotators picked about 14 of roughly 17,000 channels a call spans (2.04 m spacing, median 35 km aperture), so denser picking is the untested way to lower the noise. Source: paper/results/depth_signal_vs_noise_summary.csv and picks_per_call_summary.csv, commit ee89913.")

# ============================================================ 21 statement
add("statement",
    f"background:{ACC_BG}; color:{LIGHT}; font-family:{SANS}; padding:128px; display:flex; flex-direction:column; justify-content:center; gap:40px",
    f'<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:#A6E3DC">Where we landed</p>'
    + f'<h1 style="font-family:{SERIF}; font-size:88px; font-weight:600; line-height:1.1; color:{LIGHT}; width:1500px">Both routes to depth are closed on this dataset. What’s left is a result about geometry that travels.</h1>'
    + f'<p style="font-size:34px; line-height:1.45; color:#D6EEEA; width:1400px">Amplitude carries no depth signal. Moveout can’t separate depth from position on a monotonic arc. Sound speed was never the lever for horizontal position.</p>',
    "The one-sentence summary of Plan A plus Plan B. The negative result on this dataset is solid and pre-registered. The positive, generalisable part is the exchangeability mechanism: it says which cable layouts could make depth observable. That's the paper.")

# ============================================================ 22 divider next
divider("div-next", "Next", "From findings to a paper",
        "What the paper claims, what’s missing before we can submit, and what we need from the team.",
        "The remaining slides: the paper pitch, the gaps, the headline experiment, and the ask.")

# ============================================================ 23 paper pitch
contrib = [("Mechanism", "Depth ⇄ along-cable exchangeability on a monotonic arc, and the layouts that break it."),
           ("Pre-registered negative", "The depth gate fails: 17 / 271, 15 of them artifacts."),
           ("Sound speed", "Correction sits at 6–7% of the noise floor; free-c is a nuisance sink."),
           ("Path", "Direct arrival favoured by amplitude and, independently, by timing."),
           ("Data lessons", "Release call IDs; state depth ground truth; report σz, not a point.")]
cards = "".join(
    f'<div style="flex:1; display:flex; flex-direction:column; gap:14px; background:#FFFDF8; padding:32px; border:1px solid #DDD8CB; border-radius:16px">'
    f'<h3 style="font-size:30px; font-weight:600; color:{TEAL_T if i == 0 else INK}">{t}</h3>'
    f'<p style="font-size:24px; line-height:1.4; color:{BODY}">{d}</p></div>' for i, (t, d) in enumerate(contrib))
add("paper", LIGHT_SEC,
    header("The paper", "“When can a seafloor fibre hear depth?”")
    + p("Thesis: on the Oregon cables, source depth can’t be recovered from DAS moveout — and the reason is geometric, so it tells you how to lay a cable that could.", 32, INK, "width:1500px")
    + f'<div style="display:flex; flex-direction:row; gap:24px">{cards}</div>'
    + footer("Target: DCLDE 2027, Cornell · 2–6 Aug 2027. Submission deadline: [TBD]."),
    "Working title. Five contributions; lead with the mechanism because it generalises past this dataset. Everything else is supporting evidence and methods lessons. Venue: DCLDE 2027 at the Yang Center, Cornell. The submission deadline isn't known yet.")

# ============================================================ 24 gaps
add("gaps", LIGHT_SEC,
    header("Before we submit", "Seven gaps, in priority order")
    + table([["Work item", "Why it matters"],
             ["1 · Synthetic geometry sweep: σz vs. depth monotonicity × bearing", "Turns the 3-point fixture into the paper’s key figure"],
             ["2 · Re-run the gate on a −2000 m grid for all 114 floor-bound calls", "The limitation reviewers will hit first"],
             ["3 · Commit the numbers that aren’t in a table yet", "Grid probe, 96 ms floor, error-in-variables bound, tolerance sweep (depth signal vs noise: done, ee89913)"],
             ["4 · Second sound-speed product (WOA23)", "Our own spec requires two; every profile result rests on one"],
             ["5 · Organizers: call IDs, depth ground truth, how Z is scored. Emailed 2026-10-07, awaiting reply", "Decides whether the L-2 depth score is even meaningful"],
             ["6 · Optional: raw strain segment to test the denoiser", "Would rescue the amplitude result, or retire it"],
             ["7 · Test whether denser-channel picking gets pick noise toward ~12 ms", "The gap between 13 ms of depth signal and 98 ms of noise is about 8×; untested whether picks can close it"]],
            [58, 42], size=26),
    "Priority order is my recommendation. Item 1 is the difference between 'a negative result on one dataset' and 'a design rule'. Item 2 answers the obvious reviewer objection that the grid, not the data, stopped 114 calls. Item 3: several numbers in our findings docs are marked not-table-backed. Item 4: our spec's own two-product bar is unmet. Item 5: if challenge ground truth is the baseline's fixed-depth output, the Z component of L-2 can't be scored. Item 6: public raw data DOI 10.58046/5J60-FJ89. Item 7 (new): annotators picked about 14 of roughly 17,000 channels a call spans, so picking from the dense channels is the untested way to lower the noise; Plan A found a structured 76-82 ms range trend in the residual, so it may not be enough. Item 5 update: the email to the challenge lead went out on 2026-10-07; no reply yet. Item 3 update: the depth-signal-vs-noise numbers are now committed (commit ee89913).")

# ============================================================ 25 headline experiment sketch
cells = []
for i in range(10):          # x: bearing difference
    for j in range(7):       # y: depth monotonicity (top = V-shaped / non-monotonic)
        s = min(1.0, (i / 9) * 0.85 + ((6 - j) / 6) * 0.6)
        # interpolate coral(bad) -> teal(good)
        c0, c1 = (0xD9, 0x71, 0x3F), (0x0F, 0x9A, 0x92)
        col = "#%02X%02X%02X" % tuple(int(a + (b - a) * s) for a, b in zip(c0, c1))
        cells.append(f'<rect x="{i*72+2}" y="{j*72+2}" width="68" height="68" rx="6" fill="{col}" />')
sweep_svg = (f'<svg aria-label="Sketch of a heatmap: depth uncertainty is poor for a single monotonic cable and improves as the bearing difference between cables grows or the depth profile becomes non-monotonic" width="724" height="508" viewBox="0 0 724 508" xmlns="http://www.w3.org/2000/svg">'
             + "".join(cells) + '</svg>')
add("sweep", LIGHT_SEC,
    header("The headline experiment", "Map where depth becomes observable")
    + '<div style="display:flex; flex-direction:row; gap:72px; align-items:center; flex:1">'
    + '<div style="position:relative; width:724px; height:620px">'
    + f'<p style="position:absolute; left:0px; top:0px; width:724px; font-size:24px; color:{BODY}">↑ depth along the cable more V-shaped</p>'
    + f'<div style="position:absolute; left:0px; top:52px; width:724px; height:508px">{sweep_svg}</div>'
    + f'<p style="position:absolute; left:0px; top:576px; width:724px; font-size:24px; color:{BODY}; text-align:right">bearing difference between cables →</p>'
    + '</div>'
    + '<div style="flex:1; display:flex; flex-direction:column; gap:28px">'
    + p("Simulate synthetic calls on families of cable layouts, add pick noise at this dataset’s level, and measure σz.", 30, INK)
    + p("Output: a map of where σz ≤ 100 m is achievable, with the Oregon cables placed on it.", 28)
    + p("Cheap: forward model, profiler and noise level already exist.", 28, TEAL_T, "font-weight:600")
    + '</div></div>'
    + footer("Sketch of the proposed figure — not data. Colour: teal = depth observable, coral = not."),
    "Proposed key figure for the paper. Axes: how different the two cables' bearings are, and how non-monotonic the depth-along-cable is. Colour would be the fraction of synthetic calls reaching sigma_z <= 100 m at the dataset's ~96 ms pick noise. The sketch shows the expected shape only; nothing on it is measured. All the pieces exist: forward.travel_time, moveout.depth_profile and marginal_sigma_z, and the per-call noise distribution.")

# ============================================================ 26 ask
asks = [("Owner for the geometry sweep", "the paper’s key figure; ~all tooling exists"),
        ("Someone to follow up with the organizers", "emailed 2026-10-07: call-ID mapping, depth ground truth, how Z is scored; awaiting reply"),
        ("Two reviewers for the draft", "physics check on the propagation claims; stats check on the gate"),
        ("A go / no-go on DCLDE 2027", "submit this as its own paper, alongside our challenge entry")]
ask_html = "".join(
    f'<div style="display:flex; flex-direction:row; gap:28px; align-items:start">'
    f'<x-icon name="Check" style="color:{D_ACC}; width:44px; height:44px"></x-icon>'
    f'<div style="flex:1; display:flex; flex-direction:column; gap:4px"><h3 style="font-size:36px; font-weight:600; color:{D_TEXT}">{t}</h3>'
    f'<p style="font-size:28px; line-height:1.4; color:{D_BODY}">{d}</p></div></div>' for t, d in asks)
add("ask", DARK_SEC,
    header("The ask", "What we need from the team", dark=True)
    + f'<div style="display:flex; flex-direction:column; gap:36px">{ask_html}</div>'
    + footer("Everything shown is reproducible from committed configs on the sound-speed-research branch.", dark=True),
    "Close. Four concrete asks. The organizer email has been sent (2026-10-07), so that ask is now about following up on the reply. The code, configs, result tables and findings docs are all on the sound-speed-research branch: paper/docs/sound-speed-feasibility.md, das-finwhale-soundspeed-plan-a-findings.md, das-finwhale-transmission-loss-findings.md, das-finwhale-plan-b-gate-findings.md.")

# ------------------------------------------------------------ write
order = [s for s, _ in slides]
for sid, html in slides:
    (SL / f"{sid}.html").write_text(html, encoding="utf-8")
deck = {
    "v": 4,
    "createdOnFiles": {"v": 1, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    "title": "Can a Fibre Hear Depth?",
    "order": order,
    "sections": {
        "s-idea": {"description": "The idea: bring real ocean sound speed into DAS localization, and why it might unlock depth", "start": "cover"},
        "s-a": {"description": "Plan A: sound speed and propagation path — what we measured", "start": "div-a"},
        "s-b": {"description": "Plan B: the pre-registered depth gate, and why it fails", "start": "div-b"},
        "s-next": {"description": "From findings to a paper: claims, gaps, headline experiment, the ask", "start": "div-next"},
    },
    "faces": {
        "source-serif-4": {"family": "Source Serif 4", "href": "https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
        "ibm-plex-sans": {"family": "IBM Plex Sans", "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"},
    },
    "cover": "cover",
    "designSystems": [],
}
(ROOT / "project" / "deck.json").write_text(json.dumps(deck, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(order), "slides:", ", ".join(order))
