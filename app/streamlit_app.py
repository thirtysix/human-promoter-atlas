"""Human Promoter Atlas — Streamlit app entry (multipage)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

# Make `app.*` importable when running `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.lib import nav  # noqa: E402
from app.tabs import (aggregate, programs, archetypes, go_search,  # noqa: E402
                       transcript, compare, tf, tf_network, methods)


DEFAULT_GENE = "GAPDH"
DEFAULT_TF   = "CTCF"


def _seed_from_query_params() -> None:
    """Read ?gene= / ?tf= / ?program= and seed selectbox state. Sets sane
    defaults so first-time visitors land on something interesting."""
    from app.lib import db
    qp = st.query_params

    valid_genes = set(db.list_genes())
    gene = qp.get("gene", "")
    if "tx_gene_select" not in st.session_state:
        st.session_state["tx_gene_select"] = (
            gene if gene in valid_genes else DEFAULT_GENE
        )

    valid_tfs = set(db.list_tfs())
    tf_q = qp.get("tf", "")
    if "tf_select" not in st.session_state:
        st.session_state["tf_select"] = (
            tf_q if tf_q in valid_tfs else DEFAULT_TF
        )

    # Seed the Programs selectbox DIRECTLY. This used to write a separate
    # `preselected_program` key that only _sync_query_params ever read -- it
    # went straight back into the URL and never reached the widget, so
    # ?program=7 rendered program 1 under a URL that still claimed 7. One key,
    # one meaning. The bound was also a literal 1..10 from the k=10 build,
    # which made programs 11..140 unbookmarkable; it now comes from the data.
    program = qp.get("program", "")
    if program and "prog_pick" not in st.session_state:
        try:
            p = int(program)
        except (ValueError, TypeError):
            p = None
        if p is not None and p in set(db.get_genome_programs()["program"]):
            st.session_state["prog_pick"] = p

    go_q = qp.get("go", "")
    if go_q and "go_search_preset" not in st.session_state:
        # Accept either the raw term (GOBP_X) or the pretty form ("x y").
        pretty = go_q.replace("GOBP_", "").replace("_", " ").lower()
        st.session_state["go_search_preset"] = pretty


def _sync_query_params() -> None:
    """Write current selectbox state back to ?gene= / ?tf= / ?program= so the
    URL is a permalink to the active view."""
    qp = st.query_params
    g = st.session_state.get("tx_gene_select") or ""
    t = st.session_state.get("tf_select") or ""
    p = st.session_state.get("prog_pick")
    new = {}
    if g and g != DEFAULT_GENE:
        new["gene"] = g
    if t and t != DEFAULT_TF:
        new["tf"] = t
    if p is not None:
        new["program"] = str(int(p))
    current = {k: qp.get(k, "") for k in ("gene", "tf", "program")}
    if any(new.get(k, "") != current.get(k, "")
           for k in ("gene", "tf", "program")):
        qp.clear()
        for k, v in new.items():
            qp[k] = v


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _build_version() -> str:
    """Short git SHA + ISO commit date for the current build.

    Prefers `app/_build_info.txt` (written at deploy time, baked into the
    container image — no runtime git dependency). Falls back to a live
    `git rev-parse` for dev. Returns 'unknown' if neither path resolves.
    """
    app_dir = Path(__file__).resolve().parent
    info = app_dir / "_build_info.txt"
    if info.exists():
        return info.read_text().strip()
    try:
        repo = app_dir.parent
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        date = subprocess.check_output(
            ["git", "-C", str(repo), "log", "-1", "--date=short",
             "--pretty=format:%cd"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return f"{sha} · {date}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


_CHROME_CSS = """
<style>
/* Slim wordmark used on non-landing pages, in place of st.title. */
.hpa-wordmark {
    font-size: 1.05rem;
    font-weight: 600;
    color: #334155;
    letter-spacing: 0.01em;
    margin: 0.25rem 0 0.75rem 0;
}
/* Make the top nav strip read as primary navigation now that emoji
   prefixes are stripped: heavier weight on the active tab + a thin
   accent underline. The Streamlit nav uses st-emotion-cache-* classes
   that change between releases, so target by role + aria-current. */
[data-testid="stHeader"] nav a,
header nav a {
    font-weight: 500;
}
[data-testid="stHeader"] nav a[aria-current="page"],
header nav a[aria-current="true"],
header nav a[aria-current="page"] {
    font-weight: 700;
    color: #0F766E !important;
    border-bottom: 2px solid #0F766E;
}

/* Strengthen bordered containers so cards read as clearer groups. */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stContainer"][data-bordered="true"] {
    border-color: #cbd5e1 !important;  /* slate-300 */
}
</style>
"""


def _inject_chrome_css() -> None:
    st.markdown(_CHROME_CSS, unsafe_allow_html=True)


def _render_footer() -> None:
    st.divider()
    st.caption(
        f"Human Promoter Atlas · build `{_build_version()}`"
    )


def main() -> None:
    st.set_page_config(
        page_title="Human Promoter Atlas",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "About": "Human Promoter Atlas — TF binding programs at canonical "
                     "human TSSs. See the Methods page for data sources and "
                     "parameters."
        },
    )

    # Build the page registry once per script run; nav.goto() reads from it.
    pages = {
        "aggregate":  st.Page(aggregate.render,  title="Aggregate",
                                url_path="aggregate", default=True),
        "programs":   st.Page(programs.render,   title="Programs & Modules",
                                url_path="programs"),
        "archetypes": st.Page(archetypes.render, title="Archetypes",
                                url_path="archetypes"),
        "go_search":  st.Page(go_search.render,  title="GO search",
                                url_path="go"),
        "transcript": st.Page(transcript.render, title="Per-transcript",
                                url_path="transcript"),
        "compare":    st.Page(compare.render,    title="Compare",
                                url_path="compare"),
        "tf":         st.Page(tf.render,         title="Per-TF",
                                url_path="tf"),
        "tf_network": st.Page(tf_network.render, title="TF network",
                                url_path="tf-network"),
        "methods":    st.Page(methods.render,    title="Methods",
                                url_path="methods"),
    }
    nav.PAGES = pages

    _seed_from_query_params()

    pg = st.navigation(list(pages.values()), position="top")
    _inject_chrome_css()
    try:
        current_path = pg.url_path
    except AttributeError:
        current_path = ""
    if current_path in ("", "aggregate"):
        st.title("Human Promoter Atlas")
        st.caption(
            "TF binding programs at canonical protein-coding TSSs of the "
            "human genome (Ensembl GRCh38.114). Built from chip-atlas peaks "
            "across ≈1,300 TFs and ≈19,700 TSSs."
        )
    else:
        st.markdown(
            "<div class='hpa-wordmark'>Human Promoter Atlas</div>",
            unsafe_allow_html=True,
        )
    pg.run()
    _render_footer()
    _sync_query_params()


if __name__ == "__main__":
    main()
