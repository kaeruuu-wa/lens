"""Lens - understand documents faster with AI-powered text analysis and summarization."""

import hashlib
import html

import plotly.express as px
import streamlit as st

from utils.pdf_reader import PDFReadError, extract_pdf_text
from utils.sentiment import SentimentError, analyze_sentiment
from utils.summarizer import (
    DEFAULT_MODEL_LABEL,
    LENGTH_SETTINGS,
    SUMMARY_MODELS,
    SummaryError,
    summarize_document,
)
from utils.text_analysis import clean_text, compute_stats, extract_keywords, top_words

# ---------- Color palette: change PALETTE to switch the whole look ----------
# Options: "Indigo", "Teal", "Ocean", "Rose", "Amber"
PALETTE = "Indigo"

PALETTES = {
    #            main color   light tint   mid tint    dark text
    "Indigo": {"accent": "#4F46E5", "soft": "#EEF0FF", "mid": "#B4B8F4", "dark": "#3730A3"},
    "Teal":   {"accent": "#0D9488", "soft": "#E3F5F2", "mid": "#8FD0C9", "dark": "#0B6B63"},
    "Ocean":  {"accent": "#2563EB", "soft": "#E8F0FE", "mid": "#A5C0F7", "dark": "#1E40AF"},
    "Rose":   {"accent": "#E11D48", "soft": "#FFE9EE", "mid": "#F5A3B5", "dark": "#9F1239"},
    "Amber":  {"accent": "#D97706", "soft": "#FFF3DC", "mid": "#F2C888", "dark": "#92400E"},
}
P = PALETTES[PALETTE]
ACCENT, SOFT, MID, DARK = P["accent"], P["soft"], P["mid"], P["dark"]

INK = "#1B2437"
MUTED = "#6B7280"
LINE = "#E6E9EF"
MIN_WORDS = 40

st.set_page_config(page_title="Lens", page_icon="🔍", layout="wide")

# ---------- Styling: one typeface, soft borders, lots of white space ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {{ font-family: 'Manrope', sans-serif !important; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1020px; }}
[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {LINE}; }}

.lens-brand {{ display:flex; align-items:center; gap:.9rem; margin-bottom:.4rem; }}
.lens-mark {{
    width: 38px; height: 38px; border-radius: 50%;
    border: 5px solid {ACCENT}; position: relative; flex-shrink: 0;
}}
.lens-mark::after {{
    content:""; position:absolute; width: 5px; height: 15px; background:{ACCENT};
    border-radius: 3px; right: -9px; bottom: -11px; transform: rotate(-45deg);
}}
.lens-brand h1 {{ margin:0; padding:0; font-size: 2.4rem; font-weight: 800; letter-spacing:-0.03em; color:{INK}; line-height:1; }}
.lens-tag {{ color:{MUTED}; font-size:1.02rem; margin: 0 0 1.6rem 0; max-width: 560px; }}
.lens-rule {{ border:0; border-top:1px solid {LINE}; margin: 0 0 1.2rem 0; }}

.lens-h {{ font-size:1.15rem; font-weight:700; color:{INK}; margin: 2.4rem 0 .2rem 0; letter-spacing:-0.01em; }}
.lens-sub {{ color:{MUTED}; font-size:.9rem; margin: 0 0 .9rem 0; }}

.lens-stats {{
    display:grid; grid-template-columns: repeat(4, 1fr);
    border:1px solid {LINE}; border-radius:14px; background:#fff; overflow:hidden;
}}
.lens-stat {{ padding:1rem 1.2rem; border-right:1px solid {LINE}; border-bottom:1px solid {LINE}; }}
.lens-stat:nth-child(4n) {{ border-right:0; }}
.lens-stat:nth-last-child(-n+4) {{ border-bottom:0; }}
.lens-stat b {{ display:block; font-size:1.55rem; font-weight:700; color:{INK}; letter-spacing:-0.02em; }}
.lens-stat span {{ font-size:.82rem; color:{MUTED}; }}
@media (max-width: 760px) {{
    .lens-stats {{ grid-template-columns: repeat(2, 1fr); }}
    .lens-stat:nth-child(4n) {{ border-right:1px solid {LINE}; }}
    .lens-stat:nth-child(2n) {{ border-right:0; }}
    .lens-stat:nth-last-child(-n+4) {{ border-bottom:1px solid {LINE}; }}
    .lens-stat:nth-last-child(-n+2) {{ border-bottom:0; }}
}}

.lens-summary {{
    background:{SOFT}; border:1px solid {LINE}; border-left:4px solid {ACCENT};
    border-radius:4px 14px 14px 4px; padding:1.3rem 1.5rem;
    font-size:1.05rem; line-height:1.75; color:{INK};
}}
.chip {{
    display:inline-block; border:1px solid {LINE}; background:#fff; color:{INK};
    border-radius:999px; padding:.3rem .85rem; margin:.15rem .3rem .15rem 0; font-size:.9rem; font-weight:500;
}}
.chip.first {{ border-color:{ACCENT}; background:{SOFT}; color:{DARK}; }}
.pill {{ display:inline-block; border-radius:999px; padding:.35rem 1rem; font-weight:700; font-size:1rem; }}
.pill.pos {{ background:#E4F6EA; color:#166534; }}
.pill.neg {{ background:#FCE9E6; color:#B42318; }}
.pill.neu {{ background:#EEF0F4; color:#4B5563; }}
.lens-conf {{ color:{MUTED}; margin-left:.7rem; font-size:.95rem; }}

.lens-empty {{ border:1px dashed {MID}; border-radius:14px; padding:2rem 2rem 1.6rem 2rem; background:#fff; }}
.lens-empty h3 {{ margin:0 0 1rem 0; font-size:1.2rem; color:{INK}; }}
.lens-step {{ display:flex; gap:.9rem; align-items:center; margin:.7rem 0; color:{INK}; }}
.lens-step b {{
    background:{ACCENT}; color:#fff; border-radius:50%; min-width:26px; height:26px;
    display:flex; align-items:center; justify-content:center; font-size:.85rem;
}}
.stButton > button {{ border-radius:10px; font-weight:600; }}
.stButton > button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="lens-brand"><div class="lens-mark"></div><h1>Lens</h1></div>
<p class="lens-tag">Understand documents faster with AI-powered text analysis and summarization.</p>
<hr class="lens-rule">
""",
    unsafe_allow_html=True,
)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### Lens")
    st.caption("Your reading assistant")
    mode = st.radio("How do you want to add your document?", ["Paste text", "Upload PDF"])

    uploaded = None
    if mode == "Upload PDF":
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    st.divider()
    st.markdown("**Settings**")
    length = st.select_slider("Summary length", options=list(LENGTH_SETTINGS), value="Medium")
    model_label = st.selectbox(
        "Summary model",
        list(SUMMARY_MODELS),
        index=list(SUMMARY_MODELS).index(DEFAULT_MODEL_LABEL),
        help="BART Large writes better summaries. DistilBART is faster and uses less memory.",
    )
    want_sentiment = st.toggle("Include sentiment analysis", value=True)

    st.divider()
    analyze = st.button("Analyze document", type="primary", use_container_width=True)
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("summary_cache", None)
        st.rerun()

# ---------- Input area ----------
if mode == "Paste text":
    st.text_area(
        "Paste your text here",
        key="pasted_text",
        height=220,
        placeholder="Paste an article, essay, report, email, or research paper...",
    )
elif uploaded is not None:
    st.success(f"📄 Ready to analyze: **{uploaded.name}**. Hit *Analyze document* in the sidebar.")


# ---------- Analysis pipeline ----------
def run_analysis(text, source, pages, readable_pages):
    """Run every step, store everything in session_state."""
    bar = st.progress(0, text="Reading your document...")
    text = clean_text(text)
    result = {
        "source": source,
        "stats": compute_stats(text),
        "top_words": top_words(text, 15),
        "keywords": extract_keywords(text, 10),
        "pages": pages,
        "readable_pages": readable_pages,
        "summary": None,
        "summary_note": None,
        "summary_error": None,
        "length": length,
        "sentiment": None,
        "sentiment_error": None,
        "want_sentiment": want_sentiment,
    }

    # Summary (cached per text + length + model, so re-runs are instant)
    model_name = SUMMARY_MODELS[model_label]
    key = (hashlib.md5(text.encode("utf-8")).hexdigest(), length, model_name)
    cache = st.session_state.setdefault("summary_cache", {})
    bar.progress(10, text="Writing your summary. The first run loads the AI model, so give it a moment...")

    def on_progress(done, total):
        bar.progress(10 + int(70 * done / total), text=f"Summarizing... step {done} of {total}")

    if key in cache:
        result["summary"], result["summary_note"] = cache[key]
    else:
        try:
            summary, note = summarize_document(text, length, model_name, on_progress)
            cache[key] = (summary, note)
            result["summary"], result["summary_note"] = summary, note
        except SummaryError as e:
            result["summary_error"] = str(e)
        except Exception as e:  # model download problems, out of memory, etc.
            result["summary_error"] = (
                "The summary model couldn't run. If you're on Streamlit Cloud, try the "
                f"DistilBART model in the sidebar (it uses less memory). Details: {e}"
            )

    if want_sentiment:
        bar.progress(85, text="Checking the tone...")
        try:
            result["sentiment"] = analyze_sentiment(text)
        except SentimentError as e:
            result["sentiment_error"] = str(e)
        except Exception as e:
            result["sentiment_error"] = f"Sentiment analysis couldn't run. Details: {e}"

    bar.progress(100, text="Done!")
    bar.empty()
    st.session_state["result"] = result


if analyze:
    try:
        if mode == "Paste text":
            raw = st.session_state.get("pasted_text", "")
            if len(raw.split()) < MIN_WORDS:
                st.warning(f"Add a little more text first. Lens needs at least {MIN_WORDS} words to say anything useful.")
            else:
                run_analysis(raw, "Pasted text", None, None)
        else:
            if uploaded is None:
                st.warning("Upload a PDF in the sidebar first.")
            else:
                with st.spinner("Reading your PDF..."):
                    text, pages, readable = extract_pdf_text(uploaded)
                if len(text.split()) < MIN_WORDS:
                    st.warning("That PDF has very little text, so there isn't much to analyze.")
                else:
                    run_analysis(text, f"PDF: {uploaded.name}", pages, readable)
    except PDFReadError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong while analyzing your document. Please try again. ({e})")


# ---------- Results ----------
def heading(title, sub=None):
    st.markdown(f'<div class="lens-h">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="lens-sub">{sub}</div>', unsafe_allow_html=True)


def render_results(res):
    s = res["stats"]
    pages = f"{res['readable_pages']} of {res['pages']}" if res["pages"] else "n/a"
    stats = [
        (f"{s['words']:,}", "Words"),
        (f"{s['sentences']:,}", "Sentences"),
        (f"{s['characters']:,}", "Characters"),
        (f"{s['unique_words']:,}", "Unique words"),
        (f"{s['avg_words_per_sentence']:.1f}", "Words per sentence"),
        (f"~{s['reading_minutes']} min", "Reading time"),
        (pages, "PDF pages read"),
        ("PDF" if res["pages"] else "Pasted", "Source"),
    ]
    heading("Overview")
    cells = "".join(f'<div class="lens-stat"><b>{v}</b><span>{l}</span></div>' for v, l in stats)
    st.markdown(f'<div class="lens-stats">{cells}</div>', unsafe_allow_html=True)

    heading("AI summary", f"{res['length']} length")
    if res["summary"]:
        st.markdown(f'<div class="lens-summary">{html.escape(res["summary"])}</div>', unsafe_allow_html=True)
        if res["summary_note"]:
            st.caption(res["summary_note"])
        st.download_button("Download summary", res["summary"], file_name="lens_summary.txt")
    else:
        st.warning(res["summary_error"] or "No summary was generated.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        heading("Most frequent words", "Common filler words are left out.")
        words = res["top_words"]
        if words:
            counts = [c for _, c in words]
            colors = [ACCENT] + [MID] * (len(words) - 1)
            fig = px.bar(x=counts, y=[w for w, _ in words], orientation="h", text=counts)
            fig.update_traces(marker_color=colors, textposition="outside", cliponaxis=False)
            fig.update_layout(
                yaxis=dict(autorange="reversed", title=None, tickfont=dict(size=13)),
                xaxis=dict(title=None, showgrid=False, showticklabels=False),
                height=420, margin=dict(l=0, r=30, t=4, b=0), bargap=0.35,
                font=dict(family="Manrope, sans-serif", color=INK),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Not enough meaningful words to chart.")
    with right:
        heading("Keywords", "Repeated phrases first, then recurring words.")
        if res["keywords"]:
            chips = "".join(
                f'<span class="chip{" first" if i == 0 else ""}">{html.escape(k)}</span>'
                for i, k in enumerate(res["keywords"])
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("No clear keywords found.")

    if res["want_sentiment"]:
        heading("Tone")
        sent = res["sentiment"]
        if sent:
            cls = {"Positive": "pos", "Negative": "neg", "Neutral": "neu"}[sent["label"]]
            st.markdown(
                f'<span class="pill {cls}">{sent["label"]}</span>'
                f'<span class="lens-conf">{sent["confidence"] * 100:.0f}% confidence</span>',
                unsafe_allow_html=True,
            )
            st.caption(sent["note"])
        else:
            st.warning(res["sentiment_error"] or "No sentiment result.")


if "result" in st.session_state:
    st.caption(f"Showing results for **{st.session_state['result']['source']}**")
    render_results(st.session_state["result"])
else:
    st.markdown(
        """
<div class="lens-empty">
<h3>Nothing to show yet</h3>
<div class="lens-step"><b>1</b><span>Choose <i>Paste text</i> or <i>Upload PDF</i> in the sidebar.</span></div>
<div class="lens-step"><b>2</b><span>Pick a summary length.</span></div>
<div class="lens-step"><b>3</b><span>Press <i>Analyze document</i>.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )