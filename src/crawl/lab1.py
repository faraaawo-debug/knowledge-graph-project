
# Pour lancer: python lab1.py
# Dépendances : pip install httpx pandas spacy et python -m spacy download en_core_web_trf


import json
import time
import re
import httpx
import pandas as pd
import spacy

# On charge le modèle spaCy 
print("Chargement du modèle spaCy...")
nlp = spacy.load("en_core_web_trf")
print("Modèle chargé.\n")

## On travaille avec les titres car on va extraire le texte via l'API Wikipedia
TITLES=[ "Artificial_intelligence","Machine_learning", "Elon_Musk","Microsoft","Google", "France", "Barack_Obama", "OpenAI"]
HEADERS={"User-Agent": "MyProject/1.0 (student project; contact@example.com)"} # En-têtes HTTP envoyés au serveur Wikipedia pour éviter d'être bloqué (anti-bot)

# %% [1] On récupère le texte Wikipédia
def fetch_wikipedia_text(title):
    """Appelle l'API Wikipedia et retourne le texte brut de la page."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action":      "query",
        "prop":        "extracts",
        "explaintext": True, # pas de HTML, texte brut uniquement
        "titles":      title,
        "format":      "json"
    }
    try:
        # ON crée un client HTTP avec les en-têtes anti-bot
        with httpx.Client(headers=HEADERS, timeout=20) as client:
            response= client.get(url, params=params)
            response.raise_for_status()
            data=response.json()
            pages= data["query"]["pages"]
            page= next(iter(pages.values()))
            return page.get("extract", None)
    except Exception as e:
        print(f"Erreur API Wikipedia : {e}")
        return None
print("=== Crawl Wikipedia ===")
records = []

for title in TITLES:
    text = fetch_wikipedia_text(title)

    if text is None:
        print(f"  Skipped (no content) : {title}")
        continue

    word_count = len(text.split())

    if word_count >= 500: #on ne garde que les pages avec au moins 500 mots (pages suffisamment riches
        records.append({
            "url":        f"https://en.wikipedia.org/wiki/{title}",
            "title":      title,
            "word_count": word_count,
            "text":       text
        })
        print(f"  Kept    : {title} ({word_count} words)")
    else:
        print(f"  Skipped : {title} ({word_count} words)")

# Sauvegarde JSONL
with open("crawler_output.jsonl", "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nSauvegardé : crawler_output.jsonl ({len(records)} pages)\n")
print("=== Extraction d'entités (NER) ===")

texts = [r["text"] for r in records]
urls  = [r["url"]  for r in records]

entities = []

for doc, url in zip(nlp.pipe(texts, batch_size=2), urls):
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}:
            entities.append({
                "entity":      ent.text,
                "type":        ent.label_,
                "source_url":  url
            })

df_entities = pd.DataFrame(entities).drop_duplicates()
df_entities.to_csv("extracted_knowledge1.csv", index=False)

print(f"  {len(df_entities)} entités extraites.")
print(df_entities.head(10).to_string())
print("\nSauvegardé : extracted_knowledge1.csv\n")
print("=== Extraction de relations ===")

def extract_relations(doc, source_url):
    relations = []
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in {"PERSON", "ORG", "GPE", "DATE"}]
        if len(ents) < 2:
            continue

        relation = sent.root.lemma_
         # Pour les verbes trop génériques (être, avoir, faire), on a décidé de garder la forme conjuguée
        if relation in {"be", "have", "do"}:
            relation = sent.root.text
        # On crée un triplet pour chaque paire d'entités consécutive
        for i in range(len(ents) - 1):
            subject = ents[i].text.strip()
            obj     = ents[i + 1].text.strip()
            if subject != obj:
                relations.append({
                    "subject":    subject,
                    "relation":   relation,
                    "object":     obj,
                    "sentence":   sent.text.strip(), #phrase source pour traçabilité
                    "source_url": source_url
                })
    return relations

all_relations = []

for doc, url in zip(nlp.pipe(texts, batch_size=1), urls):
    all_relations.extend(extract_relations(doc, url))

# Suppression des doublons et sauvegarde
df_rel = pd.DataFrame(all_relations).drop_duplicates()
df_rel.to_csv("candidate_relations.csv", index=False)

print(f"  {len(df_rel)} relations extraites.")
print(df_rel.head(10).to_string())
print("\nSauvegardé : candidate_relations.csv")
print("\n=== Lab 1 terminé ===")
print("Fichiers générés :")
print("  - crawler_output.jsonl")
print("  - extracted_knowledge1.csv")
print("  - candidate_relations.csv")
