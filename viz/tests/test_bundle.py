import base64
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from viz.bundle import BUNDLE_VERSION, encode_f32, encode_i16, inline_module, render_html


def test_encode_f32_roundtrips():
    a = np.array([1.5, -2.25, 1e6], dtype=np.float32)
    back = np.frombuffer(base64.b64decode(encode_f32(a)), dtype="<f4")
    assert np.allclose(back, a)


def test_encode_i16_roundtrips():
    a = np.array([0, -32768, 32767], dtype=np.int16)
    back = np.frombuffer(base64.b64decode(encode_i16(a)), dtype="<i2")
    assert np.array_equal(back, a)


def test_inline_module_strips_module_syntax(tmp_path):
    src = tmp_path / "m.mjs"
    src.write_text(
        "import { x } from './other.mjs';\n"
        "export const A = 1;\n"
        "export function f() { return A; }\n",
        encoding="utf-8",
    )
    out = inline_module(src)
    assert "import" not in out
    assert not re.search(r"^\s*export\s", out, re.M)
    assert "const A = 1;" in out
    assert "function f()" in out


def test_render_html_is_self_contained(tmp_path):
    payload = {"bundleVersion": BUNDLE_VERSION, "meta": {"c": 1480.0}, "picks": {}}
    html = render_html(payload, Path("viz/template"))

    assert html.lstrip().startswith("<title") or "<title" in html[:2000]
    # No runtime fetching of any kind: the Artifact CSP blocks it.
    for forbidden in ("fetch(", "XMLHttpRequest", "import("):
        assert forbidden not in html, f"template must not reference {forbidden}"
    # Relative asset references, in either quoting style.
    assert not re.search(r"""(?:src|href)\s*=\s*['"]\.?/""", html), "no relative asset references"
    # No external origin at all: the page carries no third-party library. Covers
    # protocol-relative URLs too, which a bare https?:// pattern would miss.
    assert re.findall(r'https?://[^"\'\s)]+', html) == [], "page must reference no external URL"
    assert not re.search(r"""(?:src|href)\s*=\s*['"]//""", html), "no protocol-relative URLs"
    assert "__PAYLOAD__" not in html, "payload placeholder was not substituted"


def test_render_html_stays_under_the_page_limit():
    from viz.bundle import PAGE_LIMIT_BYTES

    payload = {"bundleVersion": BUNDLE_VERSION, "meta": {}, "big": encode_f32(np.zeros(100, np.float32))}
    html = render_html(payload, Path("viz/template"))
    assert len(html.encode("utf-8")) < PAGE_LIMIT_BYTES


def test_render_html_refuses_an_oversized_page(monkeypatch):
    # Exercise the raise, not just the happy path.
    import viz.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAGE_LIMIT_BYTES", 1000)
    with pytest.raises(ValueError, match="over the"):
        render_html({"bundleVersion": BUNDLE_VERSION}, Path("viz/template"))


def test_payload_cannot_break_out_of_the_script_block():
    # A result set's free-text field is the realistic carrier here, since the
    # contract is meant to accept third-party files.
    payload = {"bundleVersion": BUNDLE_VERSION, "notes": "</script><script>alert(1)</script>"}
    html = render_html(payload, Path("viz/template"))
    assert "</script><script>" not in html
    assert "\\u003c/script>" in html or "\\u003c/script\\u003e" in html
    # Exactly the script tags the template itself declares, and no more.
    assert html.count("<script") == html.count("</script>")


def test_page_declares_charset_and_viewport():
    # The Artifact host injects both of these, but the spec promises this same
    # file also runs from GitHub Pages and from disk, where nothing does.
    # Observed without them: served with no charset header the browser guessed
    # Latin-1 and rendered "filter callsâ€¦"; and with no viewport meta a phone
    # lays the page out at a forced 980px and scales the whole thing down to
    # illegibility. The charset must be inside the first 1024 bytes to take effect.
    html = render_html({"bundleVersion": BUNDLE_VERSION, "meta": {}}, Path("viz/template"))
    head = html[:1024].lower()
    assert 'charset="utf-8"' in head or "charset=utf-8" in head
    assert 'name="viewport"' in head
    assert "width=device-width" in head


def test_layout_does_not_let_the_canvas_force_overflow():
    # A plain `1fr` is `minmax(auto, 1fr)`, so the canvas's intrinsic width can
    # push the middle column past its share and scroll the body horizontally,
    # which the page must never do. Measured in a browser before this was pinned.
    css = (Path("viz/template") / "app.css").read_text(encoding="utf-8")
    grid = re.search(r"#app\s*\{[^}]*grid-template-columns:\s*([^;]+);", css)
    assert grid, "#app must set grid-template-columns"
    assert "minmax(0" in grid.group(1), f"middle column must be minmax(0, 1fr): {grid.group(1)}"
    # And it must stack rather than squeeze on a narrow viewport.
    assert re.search(r"@media\s*\(max-width:\s*\d+px\)", css), "needs a narrow-viewport stack"


# The validated palette. Cable hues are categorical slots 2 and 3; slot 1 (blue)
# is reserved for the sequential misfit ramp so the two encodings never share a
# hue. Checked all-pairs in both modes: CVD dE 9.2 light / 9.4 dark against a >=8
# target, normal-vision 27.6 / 26.5 against a >=15 floor. Pinned here because
# nothing else would notice the CSS drifting away from the validated values --
# which is exactly what happened once already.
VALIDATED_CABLE_HUES = {
    "light": {"--cable-n": "#eb6834", "--cable-s": "#1baf7a"},
    "dark": {"--cable-n": "#d95926", "--cable-s": "#199e70"},
}


def test_cable_hues_match_the_validated_palette():
    css = (Path("viz/template") / "app.css").read_text(encoding="utf-8")
    found = {}
    for token in ("--cable-n", "--cable-s"):
        found[token] = re.findall(rf"{token}:\s*(#[0-9a-fA-F]{{6}})", css)
        assert found[token], f"{token} not declared in app.css"

    light = {t: v[0].lower() for t, v in found.items()}
    assert light == VALIDATED_CABLE_HUES["light"], f"light cable hues drifted: {light}"

    # Dark values are stepped for the dark surface, not flipped, and are declared
    # twice (media query + data-theme scope) so the toggle wins both ways.
    for token, values in found.items():
        rest = {v.lower() for v in values[1:]}
        assert rest == {VALIDATED_CABLE_HUES["dark"][token]}, (
            f"{token} dark steps drifted or are inconsistent across the two scopes: {rest}"
        )


def test_template_app_version_matches_bundle_version():
    # Two independent literals in two files; without this they drift silently and
    # the page's own mismatch guard becomes the thing that is wrong.
    html = (Path("viz/template") / "app.html").read_text(encoding="utf-8")
    m = re.search(r"APP_BUNDLE_VERSION\s*=\s*(\d+)", html)
    assert m, "app.html must declare APP_BUNDLE_VERSION"
    assert int(m.group(1)) == BUNDLE_VERSION


def test_assembled_module_script_parses_as_javascript(tmp_path):
    # inline_module strips module syntax with line-anchored regexes. They cope with
    # today's single module, but Tasks 8-13 add six more, and a multi-line import,
    # an `export default`, or the text "export " inside a template literal would be
    # mangled into something that still builds and still passes every other test
    # while shipping a dead page. Parsing the assembled script closes that gap.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    html = render_html({"bundleVersion": BUNDLE_VERSION, "meta": {}}, Path("viz/template"))
    m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
    assert m, "template must contain an inline module script"

    script = tmp_path / "assembled.mjs"
    script.write_text(m.group(1), encoding="utf-8")
    proc = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"assembled module does not parse:\n{proc.stderr}"


def test_unlisted_lib_module_is_rejected(tmp_path):
    # A module added under lib/ but forgotten in LIB_ORDER would silently not be
    # inlined, and the page would fail at runtime with an undefined function.
    import shutil

    tpl = tmp_path / "template"
    shutil.copytree("viz/template", tpl)
    (tpl / "lib" / "orphan.mjs").write_text("export const ORPHAN = 1;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not in LIB_ORDER"):
        render_html({"bundleVersion": BUNDLE_VERSION}, tpl)


def test_every_listed_lib_module_is_inlined():
    from viz.bundle import LIB_ORDER

    html = render_html({"bundleVersion": BUNDLE_VERSION}, Path("viz/template"))
    for name in LIB_ORDER:
        p = Path("viz/template/lib") / name
        if not p.exists():
            continue
        lines = [ln for ln in inline_module(p).splitlines() if len(ln.strip()) > 12]
        assert lines, f"{name} produced no inlinable content"
        for ln in lines[:5]:
            assert ln in html, f"{name} was not inlined (missing {ln.strip()[:40]!r})"
