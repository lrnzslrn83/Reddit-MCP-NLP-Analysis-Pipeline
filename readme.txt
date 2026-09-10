MCP Servers – Reddit Analysis Pipeline

Questa cartella contiene i **server MCP** utilizzati nel progetto.

## Architettura generale

L’architettura è composta da **4 server MCP**, ciascuno responsabile di una fase specifica della pipeline:

1. **Data Retrieval / Loading**
2. **Text Analysis (Cleaning, Wrangling)**
3. **Visualization / Aggregation**
4. **Knowledge Graph (Neo4j, opzionale)**

Claude agisce come orchestratore: seleziona e invoca i tool dei server, raccoglie gli output intermedi e genera un report finale basato su misure calcolate e verificabili.

---

## Pipeline dei dati

La pipeline implementata segue le seguenti fasi:

1. **Data Loading / Retrieval**
   - Recupero dei post da Reddit a partire da un topic.
   - Salvataggio dei dati grezzi in formato CSV.

2. **Cleaning**
   - Gestione di testi nulli o vuoti.
   - Normalizzazione dei campi minimi necessari all’analisi.

3. **Wrangling**
   - Arricchimento del dataset con feature NLP:
     - score/label di sentiment
     - score/label di tossicità
   - Preparazione dei dati per analisi e visualizzazione.

4. **Aggregation**
   - Calcolo di statistiche descrittive (distribuzioni, percentuali, confronti tra subreddit).
   - Produzione di output aggregati utilizzati nei grafici e nel report finale.

5. **Knowledge Graph**
   - Trasformazione del dataset analizzato in nodi e relazioni.
   - Caricamento in Neo4j per esplorazione e query strutturate.

---

## Output generati

L’esecuzione completa della pipeline produce:
- Dataset CSV (grezzo e analizzato)
- Visualizzazioni (PNG)
- Wordcloud
- Knowledge Graph interrogabile (Neo4j)

---

## Riproducibilità

Ogni server è progettato per essere indipendente e riusabile.  
L’intera pipeline può essere rieseguita su topic diversi senza modificare l’architettura, garantendo modularità e scalabilità.
