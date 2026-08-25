"""Capture a Streamlit page as a series of viewport screenshots.

A single element screenshot of a tall page comes back with white voids: plotly
only paints what is near the viewport, and screenshotting the container does
not force it to paint the rest. Scrolling first does not fix it either — the
paint is discarded again once the region leaves the viewport.

So: scroll to each offset and take a VIEWPORT screenshot there. Each strip is
guaranteed to show what a real visitor at that scroll position sees, which is
also the honest unit for a UX audit.

Usage:
    python tools/pw_strips.py http://127.0.0.1:8600/tf --out tf --strips 4
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "docs" / "ux-audit"
MAIN = "[data-testid=stMainBlockContainer]"
SCROLLER = "section[data-testid=stMain]"


def strips(url: str, out_name: str, n_strips: int = 4, width: int = 1872,
           height: int = 1200, settle_ms: int = 15000) -> list[Path]:
    out_dir = OUT_ROOT / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": width, "height": height}).new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector(MAIN, timeout=60_000)
        page.wait_for_timeout(settle_ms)
        total = page.evaluate(
            f"()=>{{const m=document.querySelector('{MAIN}');return m?m.scrollHeight:0;}}")
        span = max(1, total - height)
        for i in range(n_strips):
            y = int(span * i / max(1, n_strips - 1)) if n_strips > 1 else 0
            page.evaluate(
                f"() => {{ document.querySelector('{SCROLLER}').scrollTop = {y}; }}")
            page.wait_for_timeout(2500)
            out = out_dir / f"{out_name}-s{i+1}.png"
            page.screenshot(path=str(out))
            written.append(out)
            print(f"  {out.name}  scrollTop={y}")
        browser.close()
    print(f"{out_name}: page {total} px, {n_strips} strips")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strips", type=int, default=4)
    ap.add_argument("--settle", type=int, default=15000)
    args = ap.parse_args()
    strips(args.url, args.out, args.strips, settle_ms=args.settle)


if __name__ == "__main__":
    main()
