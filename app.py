#!/usr/bin/env python3
"""
app.py — Streamlit UI for the Multimodal Search & RAG Assistant.

Bridges to the Julia MiniVDB backend via juliacall.
Falls back to a fully-functional mock mode when Julia is unavailable.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ============================================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# JULIA BRIDGE INITIALISATION
# ============================================================================

JL_AVAILABLE = False


def _init_julia() -> bool:
    """Try to load the Julia MiniVDB module. Returns True on success."""
    global JL_AVAILABLE
    try:
        from juliacall import Main as jl  # type: ignore[import-untyped]

        project_dir = str(Path(__file__).resolve().parent)
        jl.seval(f'using Pkg; Pkg.activate("{project_dir}")')
        jl.seval(f'push!(LOAD_PATH, "{project_dir}/src")')
        jl.seval("using MiniVDB")
        jl.seval("MiniVDB.init()")
        JL_AVAILABLE = True
        return True
    except Exception as exc:
        st.warning(
            "⚠️ Julia backend not available — running in **Mock Mode**.\n\n"
            f"Details: `{exc}`\n\n"
            "All features work normally; responses are simulated. "
            "Install Julia + `juliacall` for the real engine."
        )
        JL_AVAILABLE = False
        return False


# ============================================================================
# MOCK BACKEND (used when Julia is unavailable)
# ============================================================================

class MockMiniVDB:
    """In-memory mock of the MiniVDB bridge API for demo / offline use."""

    def __init__(self) -> None:
        self.texts: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest(), 16)

    @staticmethod
    def _sim_score(query: str, candidate: str) -> float:
        """Simple Jaccard-like word-overlap score for mock retrieval."""
        q_words = set(query.lower().split())
        c_words = set(candidate.lower().split())
        if not q_words or not c_words:
            return 0.0
        intersection = q_words & c_words
        union = q_words | c_words
        return len(intersection) / len(union)

    # ---- public API (mirrors Julia bridge) ---------------------------------

    def insert_text(self, text: str) -> bool:
        if not text.strip():
            return False
        self.texts.append({"text": text})
        return True

    def insert_image(self, file_path: str) -> bool:
        if not file_path.strip():
            return False
        fname = os.path.basename(file_path)
        self.images.append({
            "path": file_path,
            "filename": fname,
            "description": f"Image: {fname}",
        })
        return True

    def query_rag(
        self, query_data: str, query_type: str, top_k: int, api_key: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Mock RAG pipeline: keyword-match retrieval + simulated LLM answer."""

        # ---- retrieve ------------------------------------------------------
        scored: list[dict[str, Any]] = []

        for t in self.texts:
            s = self._sim_score(query_data, t["text"])
            if s > 0.0:
                scored.append({
                    "type": "text",
                    "content": t["text"],
                    "score": round(s, 4),
                })

        for img in self.images:
            # Match against filename + description
            haystack = img["filename"] + " " + img["description"]
            s = self._sim_score(query_data, haystack)
            # Boost images slightly so they appear in results
            s = min(s * 1.5 + random.uniform(0.01, 0.05), 1.0)
            if s > 0.0:
                scored.append({
                    "type": "image",
                    "content": img["path"],
                    "filename": img["filename"],
                    "description": img["description"],
                    "score": round(s, 4),
                })

        # Ensure at least some results (fallback — return everything, low score)
        if not scored:
            for t in self.texts:
                scored.append({
                    "type": "text",
                    "content": t["text"],
                    "score": round(random.uniform(0.05, 0.15), 4),
                })
            for img in self.images:
                scored.append({
                    "type": "image",
                    "content": img["path"],
                    "filename": img["filename"],
                    "description": img["description"],
                    "score": round(random.uniform(0.05, 0.15), 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        # ---- generate mock answer ------------------------------------------
        answer = self._mock_answer(query_data, top, bool(api_key))
        return answer, top

    @staticmethod
    def _mock_answer(
        query: str, contexts: list[dict[str, Any]], has_api_key: bool
    ) -> str:
        disclaimer = (
            ""
            if has_api_key
            else "*(This is a simulated answer — set your `DEEPSEEK_API_KEY` for live LLM generation.)*\n\n"
        )
        n = len(contexts)
        if n == 0:
            return f"{disclaimer}No relevant documents found for: \"{query}\""

        lines = [disclaimer, f"Based on the retrieved context, here is a grounded answer:\n"]
        for i, ctx in enumerate(contexts):
            tp = ctx["type"]
            sc = ctx["score"]
            if tp == "text":
                snippet = ctx["content"][:80]
                lines.append(f"- **Source [{i+1}]** (score={sc:.4f}, text): _{snippet}..._")
            else:
                lines.append(
                    f"- **Source [{i+1}]** (score={sc:.4f}, image): "
                    f"📷 `{ctx.get('filename', '?')}` — {ctx.get('description', '')}"
                )

        lines.append("")
        lines.append(
            "The top-ranked context directly addresses the query. "
            "In a production deployment, a large language model would synthesise "
            "a detailed, cited response from these retrieved chunks."
        )
        return "\n".join(lines)


# ============================================================================
# INITIALISE BACKEND (once per session)
# ============================================================================

@st.cache_resource
def get_backend() -> tuple[bool, Any]:
    """Return (julia_available, backend_instance). Cached across reruns."""
    ok = _init_julia()
    if ok:
        from juliacall import Main as jl  # type: ignore[import-untyped]
        return True, jl
    return False, MockMiniVDB()


# ============================================================================
# CUSTOM CSS
# ============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
        }
        /* Primary button */
        div.stButton > button[kind="primary"] {
            font-size: 1.1rem;
            padding: 0.75rem 2rem;
        }
        /* Sidebar section headers */
        section[data-testid="stSidebar"] h3 {
            font-size: 1.05rem;
            letter-spacing: 0.02em;
        }
        /* Info box custom */
        .rag-context-box {
            border-left: 4px solid #4A90D9;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            background: rgba(74, 144, 217, 0.06);
        }
        /* Toast success */
        [data-testid="stToast"] {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR — CONTROL CENTER
# ============================================================================

def render_sidebar(backend: Any, jl_ok: bool) -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Control Center")

        # ---- API Key -------------------------------------------------------
        st.markdown("### 🔑 API Key")
        default_key = os.getenv("DEEPSEEK_API_KEY", "")
        api_key = st.text_input(
            "DeepSeek API Key",
            value=default_key,
            type="password",
            placeholder="sk-...",
            help="Used for live LLM generation. Leave blank for mock responses.",
            key="sidebar_api_key",
        )

        # ---- Top-K ---------------------------------------------------------
        st.markdown("### 🎯 Retrieval Settings")
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Number of nearest-neighbour contexts to retrieve.",
            key="sidebar_top_k",
        )

        # ---- Status badges -------------------------------------------------
        st.markdown("### 📡 System Status")
        col_a, col_b = st.columns(2)
        with col_a:
            if jl_ok:
                st.success("Julia: ON", icon="✅")
            else:
                st.warning("Julia: Mock", icon="⚠️")
        with col_b:
            # Check Qdrant health via Julia if available; assume up if Julia OK
            st.success("Qdrant: ON", icon="📦") if jl_ok else st.warning("Qdrant: Mock", icon="📦")

        st.divider()

        # ---- Ingest Data ---------------------------------------------------
        st.markdown("## 📥 Ingest Data")
        ingest_tab1, ingest_tab2 = st.tabs(["📝 Add Text", "🖼️ Add Image"])

        # -- Text tab --
        with ingest_tab1:
            new_text = st.text_area(
                "Document text",
                height=120,
                placeholder="Paste your document text here...",
                key="ingest_text",
            )
            if st.button("Submit Text", use_container_width=True, key="btn_text"):
                if not new_text.strip():
                    st.toast("⚠️ Please enter some text first.", icon="⚠️")
                else:
                    ok = _call_insert_text(backend, jl_ok, new_text.strip())
                    if ok:
                        st.toast("✅ Text document indexed successfully!", icon="✅")
                        st.session_state["ingest_text"] = ""
                    else:
                        st.toast("❌ Failed to index text.", icon="❌")

        # -- Image tab --
        with ingest_tab2:
            uploaded = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                key="ingest_image_uploader",
            )
            if uploaded is not None:
                # Preview the uploaded image
                st.image(uploaded, caption="Preview", use_container_width=True)

            if st.button("Submit Image", use_container_width=True, key="btn_image"):
                if uploaded is None:
                    st.toast("⚠️ Please upload an image first.", icon="⚠️")
                else:
                    ok = _handle_image_upload(backend, jl_ok, uploaded)
                    if ok:
                        st.toast("✅ Image indexed successfully!", icon="✅")
                        st.session_state["ingest_image_uploader"] = None
                        st.rerun()
                    else:
                        st.toast("❌ Failed to index image.", icon="❌")

        st.divider()
        st.caption(
            "Tech stack: [Julia](https://julialang.org) · CLIP · Qdrant · Streamlit\n\n"
            "Built with ❤️ using MiniVDB"
        )

    # Return sidebar state that the main area needs
    return api_key, top_k


# ============================================================================
# BACKEND CALL HELPERS
# ============================================================================

def _call_insert_text(backend: Any, jl_ok: bool, text: str) -> bool:
    """Call backend.insert_text, dispatching to Julia or mock."""
    try:
        if jl_ok:
            return bool(backend.seval(f'MiniVDB.insert_text({repr(text)})'))
        return backend.insert_text(text)
    except Exception as exc:
        st.error(f"Insert failed: {exc}")
        return False


def _handle_image_upload(backend: Any, jl_ok: bool, uploaded) -> bool:
    """Save uploaded image to a temp file, then call backend.insert_image."""
    suffix = Path(uploaded.name).suffix if uploaded.name else ".png"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        if jl_ok:
            # Julia handles encoding; just pass the file path
            return bool(backend.seval(f'MiniVDB.insert_image({repr(tmp_path)})'))
        return backend.insert_image(tmp_path)
    except Exception as exc:
        st.error(f"Image upload failed: {exc}")
        return False


def _call_query_rag(
    backend: Any, jl_ok: bool, query_data: str, query_type: str, top_k: int, api_key: str
) -> tuple[str, list[dict[str, Any]]]:
    """Call backend.query_rag, dispatching to Julia or mock."""
    if jl_ok:
        import json
        # Julia returns a Tuple{String, Vector{Dict}} — juliacall converts
        # this to a Python tuple (str, list[dict]).
        result = backend.seval(
            f'MiniVDB.query_rag({repr(query_data)}, {repr(query_type)}, '
            f'{top_k}, {repr(api_key)})'
        )
        answer, contexts = result
        # juliacall may return Julia dicts; normalise to Python dicts
        contexts_py = []
        for ctx in contexts:
            contexts_py.append({
                "type": str(ctx.get("type", "unknown")),
                "content": str(ctx.get("content", "")),
                "score": float(ctx.get("score", 0.0)),
                "filename": str(ctx.get("filename", "")),
                "description": str(ctx.get("description", "")),
                "source_image": str(ctx.get("source_image", "")),
            })
        return str(answer), contexts_py
    else:
        return backend.query_rag(query_data, query_type, top_k, api_key)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def render_main_area(backend: Any, jl_ok: bool, api_key: str, top_k: int) -> None:
    # ---- Title -------------------------------------------------------------
    st.markdown(
        """
        <h1 style="margin-bottom: 0.2rem;">🤖 Multimodal RAG Assistant Engine</h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "**Julia** `MiniVDB` × **CLIP** (`ViT-B/32`) × **Qdrant** Vector DB × **DeepSeek** LLM × **Streamlit** UI"
    )

    st.divider()

    # ---- Query Tabs --------------------------------------------------------
    tab_text, tab_image = st.tabs(["🔍 Search by Text", "🖼️ Search by Image"])

    with tab_text:
        query_text = st.text_input(
            "Enter your query",
            placeholder="e.g. a photo of an animal, quantum computing basics, ...",
            key="query_text_input",
        )

    with tab_image:
        query_image_file = st.file_uploader(
            "Upload a query image",
            type=["png", "jpg", "jpeg", "webp"],
            key="query_image_uploader",
        )
        if query_image_file is not None:
            st.image(query_image_file, caption="Query Image", width=280)

    # ---- Execute Button ----------------------------------------------------
    col_btn, col_spacer = st.columns([3, 5])
    with col_btn:
        execute = st.button(
            "🚀 Execute Multimodal Search & RAG",
            type="primary",
            use_container_width=True,
        )

    if not execute:
        # Show placeholder before first query
        st.info(
            "👆 Enter a query above and click **Execute** to run the full RAG pipeline. "
            "Retrieved contexts and the LLM-generated answer will appear here."
        )
        return

    # ---- Determine query source --------------------------------------------
    active_tab = "text"
    query_data = query_text.strip() if query_text else ""

    if not query_data:
        # Check image tab
        if query_image_file is not None:
            active_tab = "image"
            suffix = Path(query_image_file.name).suffix if query_image_file.name else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(query_image_file.getvalue())
                query_data = tmp.name
        else:
            st.warning("⚠️ Please enter a text query or upload a query image before executing.")
            return

    # ---- Execute pipeline --------------------------------------------------
    with st.spinner("🔎 Encoding query → searching Qdrant → generating answer..."):
        start = time.perf_counter()
        try:
            answer, contexts = _call_query_rag(
                backend, jl_ok, query_data, active_tab, top_k, api_key
            )
            elapsed = time.perf_counter() - start
        except Exception as exc:
            st.error(f"❌ Query pipeline failed: {exc}")
            return

    st.success(f"✅ Query completed in {elapsed:.2f}s — {len(contexts)} results retrieved.")

    # ---- Render Results ----------------------------------------------------
    st.divider()
    st.markdown("## 📊 Retrieved Contexts (Top-K)")

    if not contexts:
        st.info("No matching documents found in the knowledge base.")
    else:
        cols = st.columns(len(contexts))
        for i, (col, ctx) in enumerate(zip(cols, contexts)):
            with col:
                score = ctx.get("score", 0.0)
                st.metric(label=f"Rank #{i+1}", value=f"{score:.4f}")

                tp = ctx.get("type", "unknown")
                if tp == "text":
                    st.markdown(
                        f'<div class="rag-context-box">📝 {ctx.get("content", "")}</div>',
                        unsafe_allow_html=True,
                    )
                elif tp == "image":
                    path = ctx.get("content", "")
                    desc = ctx.get("description", "")
                    fname = ctx.get("filename", "")
                    st.markdown(
                        f'<div class="rag-context-box">📷 <code>{fname}</code><br/>{desc}</div>',
                        unsafe_allow_html=True,
                    )
                    # Try to render the actual image if the file exists
                    if path and os.path.isfile(path):
                        st.image(path, use_container_width=True)
                    elif path:
                        st.caption(f"📁 `{path}` (file not found on disk)")
                else:
                    st.text(str(ctx.get("content", "")))

    # ---- Render LLM Answer -------------------------------------------------
    st.divider()
    st.markdown("## 💬 Grounded LLM Answer")

    if answer:
        st.markdown(answer)
    else:
        st.warning("No answer generated.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    inject_css()

    jl_ok, backend = get_backend()

    # Initialise Julia bridge if available
    if jl_ok:
        try:
            backend.seval("MiniVDB.init()")
        except Exception as exc:
            st.warning(f"Julia init warning: {exc}")

    api_key, top_k = render_sidebar(backend, jl_ok)
    render_main_area(backend, jl_ok, api_key, top_k)


if __name__ == "__main__":
    main()
