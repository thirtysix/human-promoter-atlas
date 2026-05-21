"""Small UI helpers shared across tabs."""
from __future__ import annotations

import streamlit as st


def intro_card(title: str, what: str,
               objective: str, significance: str,
               icon: str = "") -> None:
    """Renders an 'About this view' card at the top of a tab.

    Three short paragraphs: what's shown, what it's for, why it matters.
    Kept narrow on purpose so a first-time visitor isn't lectured.
    """
    with st.container(border=True):
        heading = f"{icon}  {title}".strip() if icon else title
        st.markdown(f"#### {heading}")
        st.markdown(
            f"<div style='line-height:1.55'>"
            f"<b>What you're seeing.</b> {what}<br>"
            f"<b>Objective.</b> {objective}<br>"
            f"<b>Why it matters.</b> {significance}"
            f"</div>",
            unsafe_allow_html=True,
        )
