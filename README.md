# Knowledge Graph Construction, Reasoning & RAG


**Authors : Faraa Awoyemi & Lilia Benabdallah**
*ESILV, Generative AI & AI Agents, 2026*
 
---

End-to-end pipeline built on Wikipedia data:
**Web Crawling → NER → RDF Knowledge Base → Alignment → KGE → RAG**

---

## Hardware Requirements

- macOS (tested on iMac, Intel)
- RAM : 16 GB minimum
- Storage : ~2 GB for the expanded KB
- No GPU required (CPU training supported)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment
```bash
python3.9 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 4. Install Java (required for OWL reasoner HermiT)
```bash
brew install --cask temurin
```

### 5. Install Ollama and pull the model (required for Lab 4 RAG)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:1b
```

---

## How to Run

### Lab 1 — Web Crawling & NER
```bash
cd src/crawl
python lab1.py
cd ../..
```
Outputs: `crawler_output.jsonl`, `extracted_knowledge1.csv`, `candidate_relations.csv`

---

### Lab 2 — KB Construction, Alignment & Expansion
```bash
cd src/kg
cp ../crawl/extracted_knowledge1.csv .
cp ../crawl/candidate_relations.csv .
python lab2.py
cd ../..
```
Outputs: `expanded_kb.ttl`, `expanded_kb.nt`, `ontology.ttl`, `alignment.ttl`, `kb_stats.json`

⚠️ This step makes API calls to Wikidata and may take **20–40 minutes**.

---

### Lab 3 Part 1 — SWRL Reasoning
```bash
cd src/reason
python lab3_part1.py
cd ../..
```
Output: Individuals inferred as `oldPerson` printed to console (Peter=70, Marie=69).

---

### Lab 3 Part 2 — Knowledge Graph Embedding
```bash
cd src/kge
cp ../kg/expanded_kb.nt .
python lab3_part2.py
cd ../..
```
Outputs: `train.txt`, `valid.txt`, `test.txt`, `evaluation_results.csv`, `tsne_clusters.png`

⚠️ Training runs on CPU and may take **4–6 hours**.

---

### Lab 4 — RAG with RDF/SPARQL

Start Ollama (runs automatically in background on macOS):
```bash
ollama serve
```

Then run the RAG demo:
```bash
cd src/rag
cp ../kg/expanded_kb.ttl .
python lab4_rag_sparql.py
cd ../..
```

The script runs 5 predefined demo questions automatically, then switches to interactive mode.

---

## Project Structure

```
project-root/
├── src/
│   ├── crawl/
│   │   └── lab1.py              # Web crawling + NER extraction
│   ├── kg/
│   │   └── lab2.py              # KB construction + alignment + SPARQL expansion
│   ├── reason/
│   │   ├── lab3_part1.py        # SWRL reasoning with OWLReady2
│   │   └── family.owl           # OWL ontology for SWRL
│   ├── kge/
│   │   └── lab3_part2.py        # Knowledge Graph Embedding (TransE, DistMult)
│   └── rag/
│       └── lab4_rag_sparql.py   # RAG pipeline (NL→SPARQL + self-repair)
├── data/
│   └── samples/
│       ├── crawler_output.jsonl
│       ├── extracted_knowledge1.csv
│       └── candidate_relations.csv
├── kg_artifacts/
│   ├── expanded_kb.ttl
│   ├── expanded_kb.nt
│   ├── ontology.ttl
│   ├── alignment.ttl
│   ├── alignment_table.csv
│   └── kb_stats.json
├── kge_datasets/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
├── reports/
│   ├── evaluation_results.csv
│   ├── size_sensitivity.csv
│   ├── tsne_clusters.png
│   └── nearest_neighbors.json
├── README.md
├── requirements.txt
└── .gitignore
```

---

## KB Statistics

| Metric | Value |
|---|---|
| Total triplets | 242,571 |
| Entities | 198,173 |
| Relations (after filtering) | 184 |
| Aligned entities (Wikidata) | 2,032+ |

---

## Notes

- The expanded KB is large (~2 GB). If storage is an issue, use Git LFS or provide a download link.
- KGE training requires no GPU but is significantly faster with one.
- Wikidata API calls are rate-limited — do not interrupt `lab2.py` mid-run.
- Lab 4 RAG uses hardcoded SPARQL queries for reliability — automatic SPARQL generation requires a larger LLM (7B+).
