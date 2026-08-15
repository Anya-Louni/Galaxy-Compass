"""Keep the page's script block free of literal non-ASCII characters.

The single-file build strips the document head, and with it the charset
declaration, so a host that defaults to windows-1252 will mis-decode any
literal multi-byte character sitting in the JavaScript. Escaping them keeps
the source pure ASCII and the rendering identical under any charset.

HTML text outside the script block uses named entities and is checked here
rather than rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "web" / "index.html"

ESCAPES = {
    "×": "u00d7",  # multiplication sign
    "·": "u00b7",  # middle dot
    "°": "u00b0",  # degree sign
    "±": "u00b1",  # plus-minus
    "—": "u2014",  # em dash
    "–": "u2013",  # en dash
    "…": "u2026",  # ellipsis
    "→": "u2192",  # rightwards arrow
}


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    cut = src.index("<script>")
    head, js = src[:cut], src[cut:]

    stray = sorted({c for c in head if ord(c) > 127})
    if stray:
        print("non-ASCII in markup, use named entities instead: "
              + ", ".join(f"U+{ord(c):04X}" for c in stray))
        return 1

    n = 0
    for ch, code in ESCAPES.items():
        n += js.count(ch)
        js = js.replace(ch, "\\" + code)

    TARGET.write_text(head + js, encoding="utf-8", newline="")
    out = TARGET.read_text(encoding="utf-8")

    left = sum(1 for c in out if ord(c) > 127)
    doubled = out.count("\\\\u")
    print(f"escaped {n} literal characters in the script block")
    print(f"  non-ASCII remaining: {left}")
    print(f"  doubled escape sequences: {doubled}")
    return 0 if left == 0 and doubled == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
