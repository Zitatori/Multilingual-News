# app.py
# 使い方: streamlit run app.py
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="War Framing — Media Comparison", layout="wide")

# ===== Load =====
df = pd.read_parquet("out/articles_with_sentiment.parquet")
vocab = pd.read_parquet("out/vocab.parquet")

# normalize labels
for _d in (df, vocab):
    _d["lang"] = _d["lang"].astype(str).str.strip().str.lower()

langs_all = sorted(df["lang"].unique().tolist())
if not langs_all:
    st.error("No 'lang' labels found. Check CSV and analyze.py.")
    st.stop()

# ===== Colors (fixed when known, auto otherwise) =====
COLOR_MAP_FIXED = {
    "jp":"#6E9BC5",
    "west":"#4CAF50",
    "ru":"#E53935",
    "cn":"#F4B400",
    "ua":"#1E88E5",
    "cee":"#8E24AA",
    "eu":"#3949AB",
    "tr":"#00897B",
    "me":"#6D4C41",
    "in":"#EF6C00",
    "by":"#5D4037",
    "ir":"#B71C1C",
    "wire":"#607D8B",
}
PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
COLOR_MAP_INTERNAL = {l: COLOR_MAP_FIXED.get(l, PALETTE[i % len(PALETTE)]) for i, l in enumerate(langs_all)}

# ===== Display names (English + flag emoji) =====
DISPLAY_EN = {
    "jp":   "🇯🇵 Japan",
    "west": "🌎 West (US/UK/EU)",
    "ru":   "🇷🇺 Russia",
    "cn":   "🇨🇳 China",
    "ua":   "🇺🇦 Ukraine",
    "cee":  "🇵🇱 Central & Eastern Europe",   # emoji is representative; customize if you prefer
    "eu":   "🇪🇺 European Union",
    "tr":   "🇹🇷 Turkey",
    "me":   "🇶🇦 Middle East",               # change to 🇦🇪 or others if you prefer
    "in":   "🇮🇳 India",
    "by":   "🇧🇾 Belarus",
    "ir":   "🇮🇷 Iran",
    "wire": "📰 News Wires (Reuters/AP)",
}
to_display = lambda key: DISPLAY_EN.get(key, key.upper())
to_internal = {to_display(k): k for k in langs_all}

df["lang_display"] = df["lang"].map(to_display)
vocab["lang_display"] = vocab["lang"].map(to_display)

# color map keyed by display name
COLOR_MAP_DISPLAY = {to_display(k): v for k, v in COLOR_MAP_INTERNAL.items()}

# ===== UI =====
st.title("📰 War Framing — Media Comparison (All English articles)")
st.caption("Pick blocs → see sentiment & top words. Legend shows full English names with flags.")

options = sorted(df["lang_display"].unique().tolist())
sel_display = st.multiselect("Select blocs", options=options, default=options)
sel_internal = [to_internal[d] for d in sel_display]

dfv = df[df["lang"].isin(sel_internal)].copy()
vocabv = vocab[vocab["lang"].isin(sel_internal)].copy()

st.sidebar.write("Articles:", len(dfv))
if dfv.empty:
    st.warning("No articles in the selected blocs.")
    st.stop()

# ===== Sentiment: time mean =====
st.subheader("Sentiment — Positive mean over time")
daily = (
    dfv.groupby(["lang_display", "date"])[["negative", "neutral", "positive"]]
       .mean()
       .reset_index()
)
fig1 = px.line(
    daily, x="date", y="positive", color="lang_display",
    color_discrete_map=COLOR_MAP_DISPLAY, markers=True,
    title="Positive mean by bloc"
)
st.plotly_chart(fig1, use_container_width=True)

# ===== Sentiment: distribution =====
st.subheader("Sentiment — Positive distribution")
fig2 = px.box(
    dfv, x="lang_display", y="positive", color="lang_display",
    color_discrete_map=COLOR_MAP_DISPLAY, points=False,
    title="Positive score distribution by bloc"
)
st.plotly_chart(fig2, use_container_width=True)

# ===== Top words =====
st.subheader("Top words by bloc")
topn = st.slider("Top N", 5, 40, 20)
top = (
    vocabv.sort_values(["lang","freq"], ascending=[True, False])
          .groupby("lang").head(topn)
          .assign(lang_display=lambda x: x["lang"].map(to_display))
)
fig3 = px.bar(
    top, x="freq", y="word", color="lang_display", orientation="h",
    color_discrete_map=COLOR_MAP_DISPLAY, height=700,
    title=f"Top {topn} words"
)
st.plotly_chart(fig3, use_container_width=True)

# ===== Table =====
st.subheader("Sample articles")
cols = ["lang_display","published_at","source","title","url","negative","neutral","positive"]
st.dataframe(
    dfv[cols].sort_values(["lang_display","published_at"], ascending=[True, False]),
    use_container_width=True
)
