"""Page registry for the multipage app.

streamlit_app.py populates PAGES at startup; tab modules call goto() to
programmatically switch pages (e.g. quick-start buttons).
"""
from __future__ import annotations

import streamlit as st


PAGES: dict[str, "st.Page"] = {}


def goto(name: str) -> None:
    """Switch to the named page if registered. No-op otherwise."""
    page = PAGES.get(name)
    if page is not None:
        st.switch_page(page)
