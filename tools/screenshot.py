"""Minimal headless-Chrome screenshot tool for Streamlit UX audits.

Why this exists: Streamlit's SPA renders client-side, so `curl URL > foo.html`
gets you the loading shell, not the actual layout. We need a real browser to
wait for the WebSocket to drain and the page to lay out.

Usage:

    # one shot
    python tools/screenshot.py http://localhost:8501/programs

    # named output, custom viewport, longer wait
    python tools/screenshot.py http://localhost:8501/programs \\
        --out programs-baseline --viewport 1440x900 --wait 4

Output lands in docs/ux-audit/<YYYY-MM-DD>/<name>.png (gitignored).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "docs" / "ux-audit"


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s).strip("-")


def shot(url: str, out_name: str | None = None,
         viewport: str = "1872x4000", wait: float = 12.0,
         full_page: bool = True) -> Path:
    """Capture URL to docs/ux-audit/<today>/<out_name>.png. Returns path."""
    if "x" not in viewport:
        raise ValueError("viewport must be WIDTHxHEIGHT, e.g. 1440x900")
    w, h = viewport.split("x", 1)
    today = date.today().isoformat()
    out_dir = OUT_ROOT / today
    out_dir.mkdir(parents=True, exist_ok=True)

    if not out_name:
        # auto-name from URL path + query
        from urllib.parse import urlparse
        u = urlparse(url)
        out_name = _slugify(f"{u.path.strip('/') or 'home'}"
                            f"{'_' + u.query if u.query else ''}")
    out = out_dir / f"{out_name}.png"

    # Headless Chrome's `--screenshot` does NOT wait for a SPA to render —
    # `--virtual-time-budget` fast-forwards Chrome's internal clock but the
    # WebSocket handshake + Streamlit's component pipeline run on wall clock.
    # `--run-all-compositor-stages-before-draw` forces all compositor stages
    # to complete; combined with a large virtual-time-budget it usually
    # captures the rendered page. Streamlit can still need 5–10 s for heavy
    # tabs (Programs has a plotly heatmap on first load).
    cmd = [
        "google-chrome",
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={w},{h}",
        f"--virtual-time-budget={int(wait * 1000)}",
        f"--screenshot={out}",
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not out.exists():
        sys.stderr.write(res.stderr or "")
        raise RuntimeError(f"screenshot failed for {url}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--out", default=None, help="filename stem (no extension)")
    ap.add_argument("--viewport", default="1872x4000",
                    help="WIDTHxHEIGHT (default 1872x4000 — matches a 16-inch "
                         "laptop browser viewport with a tall canvas to "
                         "capture below the fold).")
    ap.add_argument("--wait", type=float, default=12.0,
                    help="seconds for Streamlit to render (default 12.0; "
                         "heavy tabs with plotly heatmaps may need 15+).")
    ap.add_argument("--no-full-page", action="store_true",
                    help="capture only the viewport, not the full scroll")
    args = ap.parse_args()
    out = shot(args.url, args.out, args.viewport, args.wait,
               full_page=not args.no_full_page)
    print(out)


if __name__ == "__main__":
    main()
