from fastmcp import FastMCP
import os
import pandas as pd
from neo4j import GraphDatabase

DATA_SALV= r"C:\Users\lsale\OneDrive\Desktop\mcp_project\analyzed_data"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "xxx"
NEO4J_PASSWORD = "xxx"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

mcp = FastMCP("neo4j_knowledge_graph")

def get_sentiment_class(score: float) -> str | None:
    """Converte il sentiment score numerico in una classe discreta. """
    
    if score is None:
        return None
    if score <= 2:
        return "Negative"
    if score >=4:
        return "Positive"
    else:
        return "Neutral"

@mcp.tool()
def build_knowledge_graph_neo4j(file:str) -> dict:
    """
    Carica un CSV analizzato in Neo4j costruendo il knowledge graph.
    - Crea nodi (:Post) e (:Subreddit)
    - Crea relazioni (:Post)-[:POSTED_IN]->(:Subreddit)
    - Nodi (:SentimentClass) collegati con [:HAS_SENTIMENT]
    """

    filename = os.path.basename(file)
    path = os.path.join(DATA_SALV,filename)
    
    if not os.path.exists(path):
        return {"error":f"{filename} non trovato in {DATA_SALV}"}
    
    df= pd.read_csv(path)
    
    required_cols = {"id","subreddit"}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        return {"error":(f"Nel file ci devono essere almeno le colonne {required_cols}, mancano {missing_cols}")}

    rows=[]
    
    for _, row in df.iterrows():
        sentiment_score = (
            float(row["sentiment_score"])
            if "sentiment_score" in df.columns and not pd.isna(row["sentiment_score"])
            else None
        )
        rows.append(
            {
                "id": str(row["id"]),
                "subreddit": str(row["subreddit"]),
                "title": str(row.get("title", "")),
                "sentiment_score": sentiment_score,
                "sentiment_class": get_sentiment_class(sentiment_score),
                "toxic_score": (
                    float(row["toxic_score"])
                    if "toxic_score" in df.columns and not pd.isna(row["toxic_score"])
                    else None
                ),
                "is_toxic": (
                    int(row["is_toxic"])
                    if "is_toxic" in df.columns and not pd.isna(row["is_toxic"])
                    else None
                ),
            }
        )

    def load_rows(tx, rows):
        tx.run(
        """
        UNWIND $rows AS r
        MERGE (s:Subreddit {name: r.subreddit})
        MERGE (p:Post {id: r.id})
        SET p.title = r.title,
            p.sentiment_score = r.sentiment_score,
            p.toxic_score = r.toxic_score,
            p.is_toxic = r.is_toxic

        FOREACH (_ IN CASE WHEN r.sentiment_class IS NULL THEN [] ELSE [1] END |
            MERGE (c:SentimentClass {label: r.sentiment_class})
            MERGE (p)-[:HAS_SENTIMENT]->(c)
        )

        MERGE (p)-[:POSTED_IN]->(s)
        """,
        rows=rows,
        )

    with driver.session() as session:
        session.execute_write(load_rows, rows)

    total = len(rows)

    return {
        "file": filename,
        "rows_imported": total,
        "message": f"Import terminato: {total} righe caricate in Neo4j da {filename}.",
    }




if __name__ == "__main__":
    mcp.run()