import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BookMatch",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0f0e0c;
    --surface:  #1a1814;
    --border:   #2e2b26;
    --gold:     #c9a84c;
    --gold-dim: #8a6f2e;
    --cream:    #f5f0e8;
    --muted:    #7a7168;
    --accent:   #e8623a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.stButton > button {
    background: var(--gold) !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #e8b85c !important;
    transform: translateY(-1px) !important;
}

[data-testid="stTabs"] button {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
}

.book-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
}
.book-card:hover {
    border-color: var(--gold-dim);
    transform: translateX(4px);
}
.book-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--border);
    position: absolute;
    top: 0.8rem;
    right: 1.2rem;
    font-weight: 900;
}
.book-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: var(--cream);
    margin-bottom: 0.2rem;
    padding-right: 2.5rem;
}
.book-author {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.score-pill {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid var(--gold-dim);
    color: var(--gold);
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
}
.score-pill.cf  { background: rgba(232,98,58,0.12); border-color: #8a3d24; color: #e8623a; }
.score-pill.pop { background: rgba(100,180,100,0.12); border-color: #3a6e3a; color: #7ecf7e; }

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: var(--cream);
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"

@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_artifacts():
    def load(name):
        with open(os.path.join(ARTIFACT_DIR, name), "rb") as f:
            return pickle.load(f)

    content_df      = load("content_df.pkl")
    tfidf_matrix    = load("tfidf_matrix.pkl")
    title_to_index  = load("title_to_index.pkl")
    book_embeddings = load("book_embeddings.pkl")
    isbn_to_index   = load("isbn_to_index.pkl")
    isbn_list       = load("isbn_list.pkl")
    popular_df      = load("popular_books_df.pkl")
    return content_df, tfidf_matrix, title_to_index, book_embeddings, isbn_to_index, isbn_list, popular_df

try:
    content_df, tfidf_matrix, title_to_index, book_embeddings, isbn_to_index, isbn_list, popular_df = load_artifacts()
    LOADED = True
except Exception as e:
    LOADED = False
    LOAD_ERROR = str(e)


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────
def get_content_recs(title, top_k=10):
    if title not in title_to_index:
        return pd.DataFrame()
    idx  = title_to_index[title]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = 0
    top  = np.argsort(sims)[::-1][:top_k]
    result = content_df.iloc[top][["Book-Title", "Book-Author"]].copy()
    result["Content-Sim"] = sims[top].round(3)
    result.index = range(1, top_k + 1)
    return result


def get_cf_recs(title, top_k=10):
    query_isbn = None
    rows = content_df[content_df["Book-Title"] == title]
    if not rows.empty:
        for isbn in rows["ISBN"]:
            if isbn in isbn_to_index:
                query_isbn = isbn
                break
    if query_isbn is None:
        return pd.DataFrame()
    q_vec = book_embeddings[isbn_to_index[query_isbn]]
    sims  = book_embeddings @ q_vec
    top   = np.argsort(sims)[::-1][1:top_k + 1]
    results = []
    for i in top:
        isbn = isbn_list[i]
        meta = content_df[content_df["ISBN"] == isbn]
        title_r  = meta.iloc[0]["Book-Title"]  if not meta.empty else "?"
        author_r = meta.iloc[0]["Book-Author"] if not meta.empty else "?"
        results.append({"Book-Title": title_r, "Book-Author": author_r, "CF-Sim": round(float(sims[i]), 3)})
    df = pd.DataFrame(results)
    df.index = range(1, top_k + 1)
    return df


def get_hybrid_recs(query_title, top_k=10, alpha=0.35, beta=0.45, gamma=0.20):
    if query_title not in title_to_index:
        return pd.DataFrame()

    # Content scores
    c_idx  = title_to_index[query_title]
    c_sims = cosine_similarity(tfidf_matrix[c_idx], tfidf_matrix).flatten()
    c_sims[c_idx] = 0

    # CF scores
    query_isbn = None
    rows = content_df[content_df["Book-Title"] == query_title]
    if not rows.empty:
        for isbn in rows["ISBN"]:
            if isbn in isbn_to_index:
                query_isbn = isbn
                break

    cf_by_title = {}
    if query_isbn:
        q_vec  = book_embeddings[isbn_to_index[query_isbn]]
        cf_all = book_embeddings @ q_vec
        for isbn, idx in isbn_to_index.items():
            meta = content_df[content_df["ISBN"] == isbn]
            if not meta.empty:
                t = meta.iloc[0]["Book-Title"]
                cf_by_title[t] = max(cf_by_title.get(t, 0), float(cf_all[idx]))

    # Popularity scores
    pop_score_map = dict(zip(popular_df["ISBN"], popular_df["weighted_score"]))
    pop_min = popular_df["weighted_score"].min()
    pop_max = popular_df["weighted_score"].max()

    scores = []
    for i, row in content_df.iterrows():
        t = row["Book-Title"]
        if t == query_title:
            continue
        c  = float(c_sims[i])
        cf = max(cf_by_title.get(t, 0.0), 0.0)
        p  = (pop_score_map.get(row["ISBN"], pop_min) - pop_min) / (pop_max - pop_min + 1e-9)
        scores.append({
            "Book-Title":   t,
            "Book-Author":  row["Book-Author"],
            "Content-Sim":  round(c,  3),
            "CF-Sim":       round(cf, 3),
            "Pop-Score":    round(p,  3),
            "Hybrid-Score": round(alpha * c + beta * cf + gamma * p, 4),
        })

    result = pd.DataFrame(scores).sort_values("Hybrid-Score", ascending=False).head(top_k)
    result.index = range(1, top_k + 1)
    return result


# ─────────────────────────────────────────────
# RENDER BOOK CARD
# ─────────────────────────────────────────────
def render_card(rank, title, author, scores):
    pills = ""
    for label, val, css_class in scores:
        pills += f'<span class="score-pill {css_class}">{label}: {val}</span>'
    st.markdown(f"""
    <div class="book-card">
        <div class="book-rank">{rank:02d}</div>
        <div class="book-title">{title}</div>
        <div class="book-author">{author}</div>
        {pills}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Book<span>Match</span></div>
    <div class="hero-sub">Hybrid Recommendation Engine &nbsp;·&nbsp; 271K Books &nbsp;·&nbsp; 1.1M Ratings</div>
</div>
""", unsafe_allow_html=True)

if not LOADED:
    st.error(f"Could not load artifacts from `{ARTIFACT_DIR}/`")
    st.code(LOAD_ERROR)
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="section-label">Configuration</span>', unsafe_allow_html=True)

    mode = st.selectbox(
        "RECOMMENDATION MODE",
        ["🔗 Hybrid (Recommended)", "📖 Content-Based", "🧠 Collaborative Filtering", "🔥 Popularity"],
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if "Hybrid" in mode:
        st.markdown('<span class="section-label">Hybrid Weights</span>', unsafe_allow_html=True)

        user_type = st.selectbox(
            "USER TYPE",
            ["Balanced (default)", "New user (cold start)", "Warm user (1–9 ratings)", "Power user (10+ ratings)", "Custom"],
        )
        presets = {
            "Balanced (default)":       (0.35, 0.45, 0.20),
            "New user (cold start)":    (0.30, 0.10, 0.60),
            "Warm user (1–9 ratings)":  (0.35, 0.25, 0.40),
            "Power user (10+ ratings)": (0.25, 0.55, 0.20),
            "Custom":                   (0.33, 0.33, 0.34),
        }
        default_a, default_b, default_g = presets[user_type]

        if user_type == "Custom":
            alpha = st.slider("α — Content",    0.0, 1.0, default_a, 0.05)
            beta  = st.slider("β — CF",         0.0, 1.0, default_b, 0.05)
            gamma = st.slider("γ — Popularity", 0.0, 1.0, default_g, 0.05)
            total = round(alpha + beta + gamma, 2)
            if abs(total - 1.0) > 0.01:
                st.warning(f"Weights sum to {total} — should be 1.0")
        else:
            alpha, beta, gamma = default_a, default_b, default_g
            col1, col2, col3 = st.columns(3)
            col1.metric("α", alpha)
            col2.metric("β", beta)
            col3.metric("γ", gamma)
    else:
        alpha, beta, gamma = 0.35, 0.45, 0.20

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    top_k = st.slider("RESULTS TO SHOW", 5, 20, 10)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Dataset Stats</span>', unsafe_allow_html=True)
    st.metric("Books in model",     f"{len(content_df):,}")
    st.metric("Total ratings",      "1.15M")
    st.metric("SVD components",     "150")
    st.metric("Explained variance", "33.3%")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🔍  Find Similar Books  ", "  🔥  Popular Right Now  "])

# ── TAB 1 ────────────────────────────────────
with tab1:
    all_titles = sorted(content_df["Book-Title"].unique().tolist())

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        query = st.selectbox(
            "SEARCH A BOOK TITLE",
            options=[""] + all_titles,
            format_func=lambda x: "Type to search..." if x == "" else x,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Recommend →")

    if query and (go or query):
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        query_meta = content_df[content_df["Book-Title"] == query].iloc[0]
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <span class="section-label">Selected Book</span>
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:var(--cream);margin-bottom:0.2rem;">
                {query}
            </div>
            <div style="color:var(--muted);font-size:0.85rem;">by {query_meta['Book-Author']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="section-label">Recommendations</span>', unsafe_allow_html=True)

        with st.spinner("Computing recommendations..."):
            if "Hybrid" in mode:
                recs = get_hybrid_recs(query, top_k=top_k, alpha=alpha, beta=beta, gamma=gamma)
                score_cols = [("Content", "Content-Sim", ""), ("CF", "CF-Sim", "cf"), ("Pop", "Pop-Score", "pop")]
            elif "Content" in mode:
                recs = get_content_recs(query, top_k=top_k)
                score_cols = [("Content Sim", "Content-Sim", "")]
            elif "Collaborative" in mode:
                recs = get_cf_recs(query, top_k=top_k)
                score_cols = [("CF Sim", "CF-Sim", "cf")]
            else:
                recs = popular_df.head(top_k)[["Book-Title","Book-Author","weighted_score"]].copy()
                recs.columns = ["Book-Title","Book-Author","Hybrid-Score"]
                score_cols = [("Pop Score", "Hybrid-Score", "pop")]

        if recs.empty:
            st.warning("No recommendations found. Try a different title or mode.")
        else:
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                pills = [(label, row[col], css) for label, col, css in score_cols if col in row]
                render_card(rank, row["Book-Title"], row["Book-Author"], pills)

        # Score breakdown chart (hybrid only)
        if "Hybrid" in mode and not recs.empty and "Content-Sim" in recs.columns:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<span class="section-label">Score Breakdown — Top 10</span>', unsafe_allow_html=True)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor("#1a1814")
            ax.set_facecolor("#1a1814")

            top10 = recs.head(10).copy()
            top10["Short"] = top10["Book-Title"].str[:35]
            x = np.arange(len(top10))
            w = 0.26

            ax.bar(x - w, top10["Content-Sim"] * alpha, w, label=f"Content (α={alpha})", color="#c9a84c", alpha=0.85)
            ax.bar(x,     top10["CF-Sim"]      * beta,  w, label=f"CF (β={beta})",       color="#e8623a", alpha=0.85)
            ax.bar(x + w, top10["Pop-Score"]   * gamma, w, label=f"Pop (γ={gamma})",     color="#7ecf7e", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(top10["Short"], rotation=35, ha="right", fontsize=7.5, color="#f5f0e8")
            ax.set_ylabel("Weighted contribution", color="#7a7168", fontsize=8)
            ax.tick_params(axis="y", colors="#7a7168")
            ax.spines[:].set_color("#2e2b26")
            ax.legend(fontsize=8, facecolor="#1a1814", labelcolor="#f5f0e8", framealpha=0.5)
            ax.set_title("How each signal contributes to hybrid score", color="#7a7168", fontsize=9, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif not query:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#3a3630;">
            <div style="font-family:'Playfair Display',serif;font-size:2rem;margin-bottom:0.5rem;">
                Search a title above
            </div>
            <div style="font-size:0.85rem;letter-spacing:0.1em;">5,400+ books available</div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 2 ────────────────────────────────────
with tab2:
    st.markdown('<span class="section-label">Top Rated Books — Bayesian Weighted Score</span>', unsafe_allow_html=True)
    st.caption("Combines average rating and vote count to prevent low-volume flukes from ranking high")

    top_n   = st.slider("Show top N books", 10, 50, 20, key="pop_slider")
    top_pop = popular_df.drop_duplicates(subset="Book-Title", keep="first").head(top_n)

    for rank, (_, row) in enumerate(top_pop.iterrows(), 1):
        pills = [
            ("Avg Rating", f"{row['avg_rating']:.2f}", ""),
            ("# Ratings",  f"{int(row['num_ratings'])}", "cf"),
            ("Score",      f"{row['weighted_score']:.3f}", "pop"),
        ]
        render_card(rank, row["Book-Title"], row["Book-Author"], pills)