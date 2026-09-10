from transformers import pipeline
import os
import pandas as pd
from fastmcp import FastMCP
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
from langdetect import detect

nltk.download("stopwords", quiet=True)

DATA_DIR = r"C:\Users\lsale\OneDrive\Desktop\mcp_project\data"
DATA_SALV = r"C:\Users\lsale\OneDrive\Desktop\mcp_project\analyzed_data"

mcp = FastMCP(name="Text_Analyzer")

pipe_sentiment = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

pipe_toxicity = pipeline(
    "text-classification",
    model="facebook/roberta-hate-speech-dynabench-r4-target"
)

@mcp.tool()
def sentiment_analysis(file: str, colonna_testo: str = "selftext") -> dict:
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato"}

    df = pd.read_csv(path)
    texts = df[colonna_testo].dropna().astype(str).tolist()

    results = pipe_sentiment(texts, truncation=True)
    scores = [int(r["label"].split()[0]) for r in results]

    s = pd.Series(scores)

    return {
        "filename": file,
        "records_analyzed": len(scores),
        "mean_sentiment": round(s.mean(), 2),
        "distribution": s.value_counts().to_dict()
    }


@mcp.tool()
def toxicity_analysis(file: str, colonna_testo: str = "selftext",) -> dict:
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato"}

    df = pd.read_csv(path)
    texts = df[colonna_testo].dropna().astype(str).tolist()

    results = pipe_toxicity(texts, truncation=True)
    toxic_labels = [r["label"].lower() for r in results]
    toxic_scores = [r["score"] for r in results]

    df["toxic_label"] = toxic_labels
    df["toxic_score"] = toxic_scores

    df["is_toxic"] = df["toxic_label"].isin(["hate", "offensive"]).astype(int)

    toxic_rate = round(100 * df["is_toxic"].mean(), 2)

    return {
        "filename": file,
        "records_analyzed": len(df),
        "toxic_rate_percent": toxic_rate,
        "toxic_posts": int(df["is_toxic"].sum()),
        "non_toxic_posts": int(len(df) - df["is_toxic"].sum()),
        "labels_distribution": df["toxic_label"].value_counts().to_dict(),
        "columns_added": ["toxic_label", "toxic_score", "is_toxic"]
    }


@mcp.tool()
def save_analyzed_posts(file: str, colonna_testo: str = "selftext") -> dict:
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato"}

    df = pd.read_csv(path)
    texts = df[colonna_testo].dropna().astype(str).tolist()

    # Sentiment
    sent_res = pipe_sentiment(texts, truncation=True)
    df["sentiment_score"] = [int(r["label"].split()[0]) for r in sent_res]

    # Tossicità
    tox_res = pipe_toxicity(texts, truncation=True)
    df["toxic_lab"] = [r["label"].lower() for r in tox_res]
    df["toxic_score"] = [r["score"] for r in tox_res]
    df["is_toxic"] = df["toxic_lab"].isin(["hate", "offensive"]).astype(int)


    base = os.path.basename(file)
    nome, ext = os.path.splitext(base)
    new_file = f"{nome}_analyzed{ext}"

    os.makedirs(DATA_SALV, exist_ok=True)
    out_path = os.path.join(DATA_SALV, new_file)
    df.to_csv(out_path, index=False)

    return {
        "message": f"Analizzati {len(df)} post. File salvato come '{new_file}'",
        "saved_file": new_file,
        "saved_path": out_path,
        "columns_added": ["sentiment_score", "toxic_lab", "toxic_score","is_toxic"],
        "mean_sentiment": round(df['sentiment_score'].mean(), 2),
        "toxic_rate_percent": round(df["is_toxic"].mean() * 100, 2)
    }
    
@mcp.tool()
def show_post(
    file: str,
    colonna_testo: str = "selftext",
    n_esempi: int = 3
) -> dict:

    """
    Mostra i post più positivi, negativi e tossici
    leggendo direttamente un file già analizzato
    (salvato in DATA_SALV).
    """
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}. Assicurati di aver usato save_analyzed_posts prima."}

    df = pd.read_csv(path)

    required_cols = {colonna_testo, "sentiment_score", "toxic_lab", "toxic_score"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"error": f"Mancano le colonne richieste nel file analizzato: {missing}"}

    if "is_toxic" in df.columns:
        tox_mask = df["is_toxic"] == 1
    else:
        tox_mask = df["toxic_lab"].isin(["hate", "offensive"])

    id_col = "id" if "id" in df.columns else None

    cols_pos = [colonna_testo, "sentiment_score"]
    if id_col:
        cols_pos = [id_col] + cols_pos
    top_pos = df.nlargest(n_esempi, "sentiment_score")[cols_pos].to_dict(orient="records")

    cols_neg = [colonna_testo, "sentiment_score"]
    if id_col:
        cols_neg = [id_col] + cols_neg
    top_neg = df.nsmallest(n_esempi, "sentiment_score")[cols_neg].to_dict(orient="records")

    df_tox = df[tox_mask]
    cols_tox = [colonna_testo, "toxic_lab", "toxic_score"]
    if id_col:
        cols_tox = [id_col] + cols_tox
    top_tox = df_tox.nlargest(n_esempi, "toxic_score")[cols_tox].to_dict(orient="records")

    return {
        "filename": file,
        "samples": {
            "top_positive": top_pos,
            "top_negative": top_neg,
            "top_toxic": top_tox
        },
        "message": f"File: {file}. Record nel file analizzato: {len(df)}. Esempi per categoria: {n_esempi}."
    }


@mcp.tool()
def word_frequency(file: str,
                   colonna_testo: str = "selftext",
                   sentiment_filter: int = 2,
                   top_n: int = 20) -> dict:

    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)
    if colonna_testo not in df.columns or "sentiment_score" not in df.columns:
        return {"error": "Servono 'sentiment_score' e la colonna testo."}

    df_filtered = df[df["sentiment_score"] <= sentiment_filter]
    texts = df_filtered[colonna_testo].dropna().astype(str).tolist()

    if not texts:
        return {"message": f"Nessun post con sentiment ≤ {sentiment_filter}."}
    
    
    stop_words = (
    set(stopwords.words("english"))
    | set(stopwords.words("italian"))
    | set(stopwords.words("french")))
    links = {"www", "https", "http", "reddit", "com", "it"}   
    stop_words|= links
    all_words = []

    for t in texts:
        t= re.sub(r"http\S+|www\.\S+", "", t)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)

    freq = Counter(all_words)
    top_words = dict(freq.most_common(top_n))

    return {
        "filename": file,
        "records_analyzed": len(df_filtered),
        "top_words": top_words,
        "message": f"Top {top_n} parole per sentiment ≤ {sentiment_filter}"
    }



@mcp.tool()
def detect_languages(file: str, colonna_testo: str = "selftext") -> dict:
    file = os.path.basename(file)

    path_data = os.path.join(DATA_DIR, file)
    path_salv = os.path.join(DATA_SALV, file)

    if os.path.exists(path_salv):
        path = path_salv
    elif os.path.exists(path_data):
        path = path_data
    else:
        return {"error": f"{file} non trovato né in {DATA_DIR} né in {DATA_SALV}"}

    df = pd.read_csv(path)
    langs = []
    for t in df[colonna_testo].fillna("").astype(str):
        try:
            langs.append(detect(t))
        except Exception:
            langs.append("unknown")
    
    total= len(langs)
    counts=Counter(langs)
    
    analisi = {
        lingua: {
            "count": counts[lingua],
            "percentage": round((counts[lingua]/total)*100,2)
        }
        for lingua in counts
    }
    return {"total_post":total,"languages":analisi}


@mcp.tool()
def compute_stats(file: str) -> dict:
    """
    Calcola statistiche riassuntive su un file già analizzato
    (salvato in DATA_SALV).
    """
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)

    stats = {
        "n_records": int(len(df)),
        "sentiment_mean": float(df.get("sentiment_score", pd.Series(dtype=float)).mean()),
        "sentiment_std": float(df.get("sentiment_score", pd.Series(dtype=float)).std())
    }

    if "is_toxic" in df.columns:
        stats["toxic_rate"] = float(df["is_toxic"].mean())
    if "toxic_score" in df.columns:
        stats["max_toxic_score"] = float(df["toxic_score"].max())

    return stats


if __name__ == "__main__":
    mcp.run()
