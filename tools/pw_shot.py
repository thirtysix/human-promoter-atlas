"""Playwright screenshot for Streamlit UX audits — the reliable capture path.

`tools/screenshot.py` (headless Chrome + --virtual-time-budget) advances
Chrome's VIRTUAL clock, but Streamlit streams plot data over a WebSocket on
WALL-CLOCK time, so on heavy tabs the shot fires while the plotly divs are
still empty. This waits on actual DOM events instead.

Two Streamlit-specific traps are handled here:
  * `document.body.scrollHeight` is 0 — Streamlit collapses body and puts the
    content in [data-testid=stMain]. Poll stMainBlockContainer instead, and
    take an ELEMENT screenshot of it; full_page=True captures only the
    viewport for the same reason.
  * Plot count alone is not "done" — a tab can paint 3 of its 5 figures and
    keep growing. Wait for BOTH the plot count and the container height to
    stop changing.

Usage:
    python tools/pw_shot.py http://127.0.0.1:8599/programs --out programs
    python tools/pw_shot.py "http://127.0.0.1:8599/transcript?gene=TP53" \
        --out transcript-TP53 --min-plots 4
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "docs" / "ux-audit"

MAIN = "[data-testid=stMainBlockContainer]"


def shot(url: str, out_name: str, width: int = 1872, min_plots: int = 1,
         settle_ms: int = 5000, timeout_s: int = 120) -> Path:
    out_dir = OUT_ROOT / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{out_name}.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(
            viewport={"width": width, "height": 1200},
            device_scale_factor=1,
        ).new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector(MAIN, timeout=60_000)
        if min_plots > 0:
            try:
                page.wait_for_selector(".js-plotly-plot", timeout=timeout_s * 1000)
            except Exception:
                sys.stderr.write(f"warn: no plotly div appeared for {url}\n")

        get_h = (f"const m=document.querySelector('{MAIN}');"
                 f"return m?m.scrollHeight:0;")
        prev_cnt = prev_h = -1
        stable = 0
        for _ in range(80):
            cnt = page.eval_on_selector_all(".js-plotly-plot", "els=>els.length")
            h = page.evaluate(f"()=>{{{get_h}}}")
            if cnt >= min_plots and cnt == prev_cnt and h == prev_h and h > 0:
                stable += 1
                if stable >= 5:
                    break
            else:
                stable = 0
            prev_cnt, prev_h = cnt, h
            page.wait_for_timeout(1200)
        page.wait_for_timeout(settle_ms)

        # Plotly only paints what is near the viewport, so a tall page
        # screenshots as bands of white where figures never rendered -- a
        # 13,305 px transcript page came back with a 2,000 px void through
        # the middle of its TF rug. Walk the scroll position down the whole
        # container (and back to the top) to force every figure to paint
        # before the element screenshot is taken.
        # Streamlit scrolls an inner element, not the window -- scrolling
        # `window` is a silent no-op and leaves the voids in place.
        scroller = ("document.querySelector('section[data-testid=stMain]')"
                    " || document.scrollingElement")
        height = page.evaluate(f"()=>{{{get_h}}}")
        step = 700
        for y in range(0, int(height) + step, step):
            page.evaluate(f"() => {{ const s = {scroller}; s.scrollTop = {y}; }}")
            page.wait_for_timeout(200)
        page.evaluate(f"() => {{ const s = {scroller}; s.scrollTop = 0; }}")
        page.wait_for_timeout(2000)
        height = page.evaluate(f"()=>{{{get_h}}}")
        target = page.query_selector(MAIN)
        target.screenshot(path=str(out))
        browser.close()
    print(f"{out}  ({height} px tall, {out.stat().st_size/1024:.0f} KB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1872)
    ap.add_argument("--min-plots", type=int, default=1,
                    help="0 for plot-less pages (GO search, TF network).")
    ap.add_argument("--settle", type=int, default=5000)
    args = ap.parse_args()
    shot(args.url, args.out, args.width, args.min_plots, args.settle)


if __name__ == "__main__":
    main()
