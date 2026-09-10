import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from fastmcp import FastMCP
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

DATA_SALV = r"C:\Users\lsale\OneDrive\Desktop\mcp_project\analyzed_data"
IMG_DIR = r"C:\Users\lsale\OneDrive\Desktop\mcp_project\visuals"
os.makedirs(IMG_DIR, exist_ok=True)

mcp = FastMCP(name="Visualizer")

def make_topic_subfolder(base_output_dir: str, filename: str) -> str:
    """
    Crea (se non esiste) una sottocartella in base al tema estratto dal nome del file.
    Esempio: reddit_tema_analyzed.csv → visuals/tema/
    """
    base_name = os.path.basename(filename)

    parts = base_name.replace(".csv", "").replace("reddit_", "").replace("_analyzed", "")
    topic_name = parts.strip() or "default"

    topic_folder = os.path.join(base_output_dir, topic_name)
    os.makedirs(topic_folder, exist_ok=True)
    return topic_folder


@mcp.tool()
def plot_sentiment_distribution(file:str) -> dict:
    """
    Crea un grafico della distribuzione dei punteggi di sentiment e restituisce
    sia il percorso dell'immagine che un riepilogo testuale leggibile da Claude.
    """
    file = os.path.basename(file)
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)
    if "sentiment_score" not in df.columns:
        return {"error": "File senza colonna sentiment_score"}
    
    sentiment_counts= df["sentiment_score"].value_counts().sort_index()
    sentiment_dict= sentiment_counts.to_dict()
    
    fig,ax= plt.subplots()
    sentiment_counts.plot(kind="bar",color="yellow",ax=ax)
    ax.set_title("Distribuzione dei punteggi di Sentiment")
    ax.set_xlabel("Valore di Sentiment")
    ax.set_ylabel("Numero di post")
    plt.tight_layout()
    topic_folder = make_topic_subfolder(IMG_DIR, file)
    out_path = os.path.join(topic_folder, f"sentiment_{os.path.basename(file).replace('.csv', '.png')}")
    plt.savefig(out_path)
    plt.close()

    
    
    total = sentiment_counts.sum()
    max_label = sentiment_counts.idxmax()
    max_value = sentiment_counts.max()
    perc = round(max_value / total * 100, 2)
    summary_text = (
        f"Analizzati {total} post. Il punteggio di sentiment più frequente è {max_label} "
        f"({perc}% dei post). Distribuzione dettagliata: {sentiment_dict}."
    )
    return {
        "message": "Grafico generato con riepilogo testuale",
        "image_path": out_path,
        "sentiment_distribution": sentiment_dict,
        "summary": summary_text
    }
    

@mcp.tool()
def generate_wordcloud(
    file: str,
    colonna_testo: str = "selftext",
    sentiment_min: int = 1,
    sentiment_max: int = 2,
    exclude_terms: list[str] = None
) -> dict:

    file = os.path.basename(file)
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)
    if colonna_testo not in df.columns or "sentiment_score" not in df.columns:
        return {"error": "File non valido: servono 'sentiment_score' e la colonna di testo."}

    df = df[(df["sentiment_score"] >= sentiment_min) & (df["sentiment_score"] <= sentiment_max)]
    if df.empty:
        return {"message": f"Nessun post con sentiment tra {sentiment_min} e {sentiment_max}."}

    text = " ".join(df[colonna_testo].dropna().astype(str).tolist())
    if not text.strip():
        return {"message": "Nessun testo disponibile per generare la WordCloud."}

    stop_words = set()
    for lang in ["english", "italian", "french"]:
        try:
            stop_words.update(stopwords.words(lang))
        except:
            pass

    links = {"www", "https", "http", "reddit", "com", "it"}
    stop_words |= links

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    if exclude_terms:
        filtered_words = [w for w in filtered_words if w not in exclude_terms]

    clean_text = " ".join(filtered_words)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="plasma",
        stopwords=stop_words
    ).generate(clean_text)

    topic_folder = make_topic_subfolder(IMG_DIR, file)
    base_name = os.path.basename(file)
    out_path = os.path.join(topic_folder, f"wordcloud_{base_name.replace('.csv', '.png')}")
    wc.to_file(out_path)

    freq = Counter(filtered_words)
    top_words = dict(freq.most_common(10))

    summary = (
        f"Generata WordCloud per {len(df)} post con sentiment tra {sentiment_min} e {sentiment_max}. "
        f"Le 5 parole più frequenti sono: {list(top_words.keys())[:5]}."
    )

    return {
        "message": "WordCloud creata con successo",
        "image_path": out_path,
        "top_words": top_words,
        "summary": summary
    }



@mcp.tool()
def generate_toxic_wordcloud(
    file: str,
    colonna_testo: str = "selftext",
    exclude_terms: list[str] = None
) -> dict:

    file = os.path.basename(file)
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}
    
    df = pd.read_csv(path)
    if colonna_testo not in df.columns or "is_toxic" not in df.columns:
        return {"error": "File non valido: serve la colonna di testo e is_toxic."}

    df_toxic = df[df["is_toxic"] == 1]
    if df_toxic.empty:
        return {"message": "Nessun post tossico trovato"}

    text = " ".join(df_toxic[colonna_testo].dropna().astype(str).tolist())
    if not text.strip():
        return {"message": "Nessun testo disponibile per generare la WordCloud."}

    stop_words = set()
    for lang in ["english", "italian", "french"]:
        try:
            stop_words.update(stopwords.words(lang))
        except:
            pass

    links = {"www", "https", "http", "reddit", "com", "it"}
    stop_words |= links

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered_words = [w for w in words if w not in stop_words]

    if exclude_terms:
        filtered_words = [w for w in filtered_words if w not in exclude_terms]

    clean_text = " ".join(filtered_words)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="inferno",
        stopwords=stop_words
    ).generate(clean_text)

    topic_folder = make_topic_subfolder(IMG_DIR, file)
    base_name = os.path.basename(file)
    out_path = os.path.join(topic_folder, f"wordcloud_toxic_{base_name.replace('.csv', '.png')}")
    wc.to_file(out_path)

    freq = Counter(filtered_words)
    top_words = dict(freq.most_common(10))

    summary = (
        f"Generata WordCloud per {len(df_toxic)} post tossici. "
        f"Le 5 parole più frequenti sono: {list(top_words.keys())[:5]}."
    )

    return {
        "message": "WordCloud tossica creata con successo",
        "image_path": out_path,
        "top_words": top_words,
        "summary": summary
    }


@mcp.tool()
def plot_posts_by_subreddit(file: str, top_n: int = 10) -> dict:
    """
    Crea un grafico che mostra quanti post provengono da ciascun subreddit.
    Riporta anche le percentuali sul totale.
    """
    file = os.path.basename(file)
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)
    if "subreddit" not in df.columns:
        return {"error": "Il file deve contenere la colonna 'subreddit'."}

    subreddit_counts = df["subreddit"].value_counts().reset_index()
    subreddit_counts.columns = ["subreddit", "post_count"]
    total_posts = subreddit_counts["post_count"].sum()
    subreddit_counts["percent"] = round(100 * subreddit_counts["post_count"] / total_posts, 2)

    top_subs = subreddit_counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_subs["subreddit"], top_subs["post_count"], color="#00b894")
    ax.set_xlabel("Numero di post")
    ax.set_ylabel("Subreddit")
    ax.set_title(f"Distribuzione dei post per i {top_n} subreddit principali")
    plt.tight_layout()

    for i, (count, perc) in enumerate(zip(top_subs["post_count"], top_subs["percent"])):
        ax.text(count + 1, i, f"{count} ({perc}%)", va='center')

    topic_folder = make_topic_subfolder(IMG_DIR, file)
    base_name = os.path.basename(file)
    out_path = os.path.join(topic_folder, f"posts_subreddit_{base_name.replace('.csv', '.png')}")
    plt.savefig(out_path)
    plt.close()

    top_sub = top_subs.iloc[0]
    summary = (
        f"Sono stati analizzati {total_posts} post provenienti da {len(subreddit_counts)} subreddit. "
        f"Il più rappresentato è '{top_sub['subreddit']}' con {top_sub['post_count']} post "
        f"({top_sub['percent']}% del totale)."
    )

    return {
        "message": "Grafico del numero di post per subreddit generato.",
        "image_path": out_path,
        "subreddit_counts": top_subs.to_dict(orient="records"),
        "summary": summary
    }


    
@mcp.tool()
def plot_sentiment_by_subreddit(file: str, top_n: int = 10) -> dict:
    """
    Crea un grafico della distribuzione media del sentiment per subreddit.
    Mostra i subreddit con più post analizzati.
    """
    file = os.path.basename(file)
    path = os.path.join(DATA_SALV, file)
    if not os.path.exists(path):
        return {"error": f"{file} non trovato in {DATA_SALV}"}

    df = pd.read_csv(path)
    if "sentiment_score" not in df.columns or "subreddit" not in df.columns:
        return {"error": "Il file deve contenere 'sentiment_score' e 'subreddit'."}

    subreddit_stats = (
        df.groupby("subreddit", as_index=False)
        .agg(
            mean_sentiment=("sentiment_score", "mean"),
            post_count=("sentiment_score", "count")
        )
        .sort_values("post_count", ascending=False)
    )

    top_subs = subreddit_stats.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_subs["subreddit"], top_subs["mean_sentiment"], color="#6c5ce7")
    ax.set_xlabel("Punteggio medio di sentiment")
    ax.set_ylabel("Subreddit")
    ax.set_title(f"Sentiment medio per i {top_n} subreddit più attivi")
    plt.tight_layout()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f"{width:.2f}", va='center')

    topic_folder = make_topic_subfolder(IMG_DIR, file)
    base_name = os.path.basename(file)
    out_path = os.path.join(topic_folder, f"sentiment_subreddit_{base_name.replace('.csv', '.png')}")
    plt.savefig(out_path)
    plt.close()

    best_sub = top_subs.iloc[0]
    worst_sub = top_subs.iloc[-1]
    summary = (
        f"Sono stati analizzati {len(df)} post provenienti da {len(subreddit_stats)} subreddit. "
        f"Il subreddit con il sentiment medio più alto è '{best_sub['subreddit']}' ({best_sub['mean_sentiment']:.2f}), "
        f"mentre quello con il sentiment più basso è '{worst_sub['subreddit']}' ({worst_sub['mean_sentiment']:.2f})."
    )

    return {
        "message": "Grafico generato con riepilogo per subreddit.",
        "image_path": out_path,
        "subreddit_stats": top_subs.to_dict(orient="records"),
        "summary": summary
    }
    



if __name__ == "__main__":
    mcp.run()
