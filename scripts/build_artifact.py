"""Inline the whole site into one self-contained HTML file.

The standalone site in web/ fetches its assets over HTTP, which is right for
a hosted deployment. A single-file build is useful for sharing somewhere
that serves exactly one document and blocks external requests.

Everything is embedded: atlas pages and sprite sheets as data URIs, JSON as
parsed literals. The page's own asset() helper already prefers
window.__ASSETS when present, so no application code changes.

Atlas tiles are rebuilt smaller for this target, because base64 inflates
binary by a third and a single document has a size ceiling that a hosted
directory does not.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT

WEB = ROOT / "web"


def data_uri(p: Path) -> str:
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=36)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--out", default=str(WEB / "artifact.html"))
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--fragment", action="store_true",
                    help="emit title+style+body only, for hosts that supply "
                         "their own document skeleton")
    a = ap.parse_args()

    build = WEB / "_artifact_assets"
    if not a.skip_export:
        print(f"exporting assets at {a.tile}px ...")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_web.py"),
             "--tile", str(a.tile), "--quality", str(a.quality),
             "--outdir", str(build)],
            check=True,
        )

    assets: dict[str, object] = {}
    total = 0
    for p in sorted(build.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(build).as_posix()
        if p.suffix == ".json":
            assets[rel] = json.loads(p.read_text())
        else:
            assets[rel] = data_uri(p)
            total += p.stat().st_size

    html = (WEB / "index.html").read_text(encoding="utf-8")
    payload = json.dumps(assets, separators=(",", ":"), allow_nan=False)
    inject = f"<script>window.__ASSETS={payload};</script>\n"

    marker = "<script>\n/* ====="
    if marker not in html:
        raise RuntimeError("could not find the main script block to inject before")
    html = html.replace(marker, inject + marker, 1)

    if a.fragment:
        # Artifact hosting wraps the supplied file in its own document
        # skeleton, so a full document would nest <html>/<body> inside
        # another one. Emit title + styles + body content instead; the
        # <title> is kept because the host reads it for the page name.
        import re

        title = re.search(r"<title>.*?</title>", html, re.S)
        style = re.search(r"<style>.*?</style>", html, re.S)
        body = re.search(r"<body[^>]*>(.*)</body>", html, re.S)
        if not (title and style and body):
            raise RuntimeError("could not split index.html into title/style/body")
        html = f"{title.group(0)}\n{style.group(0)}\n{body.group(1).strip()}\n"

    out = Path(a.out)
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}")
    print(f"  embedded binary {total / 1e6:.2f} MB  ->  single file {mb:.2f} MB")
    if mb > 15:
        print("  WARNING: over 15 MB; rerun with a smaller --tile")


if __name__ == "__main__":
    main()
