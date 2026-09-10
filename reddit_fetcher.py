import os
import pandas as pd
import praw
from fastmcp import FastMCP

reddit = praw.Reddit(
    client_id="xxx",
    client_secret="xxx",
    user_agent="xxx",
    username="xxx",
    password="xxx"
)



DATA_DIR = r"C:\Users\lsale\OneDrive\Desktop\mcp_project\data"
os.makedirs(DATA_DIR, exist_ok=True)

mcp=FastMCP(name="Reddit_Fetcher")

@mcp.tool()
def fetch_reddit_posts(topic: str, limit: int = 50) -> dict:
    """
    Scarica i post da Reddit per un topic e salva un CSV in /data.
    Restituisce il path del file e il numero di post trovati.
    """
    sf = topic.replace(" ", "_").lower()
    file_name = f"reddit_{sf}.csv"
    file_path = os.path.join(DATA_DIR, file_name)

    posts = []
    for sub in reddit.subreddit("all").search(topic, limit=limit):
        posts.append({
            "id": sub.id,
            "title": sub.title,
            "selftext": sub.selftext,
            "score": sub.score,
            "subreddit": str(sub.subreddit),
            "created_utc": sub.created_utc,
            "url": sub.url
        })

    if not posts:
        return {
            "file_path": file_path,
            "records": 0,
            "message": f"Nessun post trovato per '{topic}'"
        }

    df = pd.DataFrame(posts)
    if "title" in df.columns and "selftext" in df.columns:
        df["selftext"] = df["selftext"].replace(r'^\s*$', pd.NA, regex=True)
        df.loc[df["selftext"].isna(), "selftext"] = df["title"]

    df.to_csv(file_path, index=False)

    return {
        "file_path": file_path,
        "records": len(df),
        "message": f"Scaricati {len(df)} post su '{topic}'"
    }

@mcp.tool()
def merge_reddit_files(files: list[str], out_name: str = "merged_reddit.csv") -> dict:
    dfs = []
    merged= []
    for f in files:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            merged.append(f)
    if not dfs:
        return {"error": "Nessun file valido"}
    df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(DATA_DIR, out_name)
    df.to_csv(out_path, index=False)
    return {"message": "File uniti",
            "records": len(df),
            "output": out_name,
            "merged_files": merged}
    
@mcp.tool()
def delete_reddit_file(file: str) -> dict:
    path = os.path.join(DATA_DIR, file)
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"{file} eliminato con successo"}
    return {"error": f"{file} non esiste"}



if __name__ == "__main__":
    mcp.run()
