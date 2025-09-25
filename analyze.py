# analyze.py
# 使い方: python analyze.py
from pathlib import Path
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# ---- 1) データ読み込み ----
DATA_PATH = "data/news.csv"
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_df():
    df = pd.read_csv(DATA_PATH)
    # 統合テキスト（見出し＋本文）
    df["text_all"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    # 日付正規化
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df

# ---- 2) 英語用トークナイザ & ストップワード ----
# 記事本文は全部英語とみなす（lang列は jp/west/ru/cn/ua の“ラベル”）
STOP_EN_BASE = set("""
the of to and a in is it for on that with as at by from this be are was were been being
or an into amid among over under against during after before while through between within
will would can could should may might must do does did doing done not no nor than then
very more most much many few several any some other such new one two three four five
say says said people person group groups officials sources report reports according
""".split())

STOP_EN_NEWS = set("""
amidst amid crisis conflict situation issue issues update live latest video photo photos
breaking exclusive analysis opinion editorial comment comments view views world news
government authorities media social online reported reporting
""".split())

STOP_EN_CUSTOM = set(["with", "people","video","live","update","officials","say","says", "first", "all"])  # 追加したい語はここに足してOK

STOP = {"en": STOP_EN_BASE | STOP_EN_NEWS | STOP_EN_CUSTOM}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

def tokenize_en(txt: str):
    toks = TOKEN_RE.findall(txt.lower())
    return [t for t in toks if len(t) > 2 and t not in STOP["en"]]

def build_vocab(df: pd.DataFrame) -> pd.DataFrame:
    """lang（= jp/west/ru/cn/ua）ごとの頻出語を作る。80%以上の文書に出る語は除外。"""
    docs = df["tokens"].tolist()
    dfreq = Counter()
    for doc in docs:
        for w in set(doc):
            dfreq[w] += 1
    n_docs = max(1, len(docs))
    high_df_words = {w for w, d in dfreq.items() if d / n_docs >= 0.8}

    rows = []
    for label, sub in df.groupby("lang"):
        cnt = Counter([w for toks in sub["tokens"] for w in toks if w not in high_df_words])
        for w, n in cnt.items():
            rows.append({"lang": label, "word": w, "freq": n})
    return pd.DataFrame(rows).sort_values(["lang", "freq"], ascending=[True, False])

# ---- 3) 感情推定（多言語XLM-Rベース） ----
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# tiktoken問題の回避に use_fast=False を推奨
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
LABELS = ["negative","neutral","positive"]

@torch.inference_mode()
def sentiment_scores(texts: list[str]) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame(columns=LABELS)
    inputs = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return pd.DataFrame(probs, columns=LABELS)

# ---- 4) 実行フロー ----
def main():
    df = load_df()
    # load_df() で読み込んだ直後 or main() の最初に追記
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    print("langs in CSV:", sorted(df["lang"].unique()))
    print(df["lang"].value_counts())

    # すべて英語としてトークン化
    df["tokens"] = [tokenize_en(t) for t in df["text_all"]]

    # 感情スコア
    sent = sentiment_scores(df["text_all"].tolist())
    df = pd.concat([df.reset_index(drop=True), sent.reset_index(drop=True)], axis=1)

    # 集計用に日付
    df["date"] = df["published_at"].dt.date

    # ノイズ語除去済みの頻出語テーブル
    vocab = build_vocab(df)

    # 保存
    df.to_parquet(OUT_DIR / "articles_with_sentiment.parquet", index=False)
    vocab.to_parquet(OUT_DIR / "vocab.parquet", index=False)
    print("Saved ->", OUT_DIR / "articles_with_sentiment.parquet")
    print("Saved ->", OUT_DIR / "vocab.parquet")

if __name__ == "__main__":
    main()
