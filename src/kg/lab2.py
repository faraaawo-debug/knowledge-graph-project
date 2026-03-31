# Pour lancer : python lab2.py
# Dépendances : pip install rdflib SPARQLWrapper requests pandas tqdm
# Prérequis: avoir lancé lab1.py d'abord (génère les CSV)

import re
import json
import time
import requests
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from tqdm import tqdm
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD
from SPARQLWrapper import SPARQLWrapper, JSON


print("=== Chargement des fichiers Lab 1 ===")

entities_file = "extracted_knowledge1.csv"
relations_file= "candidate_relations.csv"

entities_df = pd.read_csv(entities_file)
relations_df= pd.read_csv(relations_file)
#normalisation des noms de colonnes (minuscules, sans espaces)
entities_df.columns = [c.strip().lower() for c in entities_df.columns]
relations_df.columns= [c.strip().lower() for c in relations_df.columns]

print(f"  Entités  : {len(entities_df)} lignes")
print(f"  Relations: {len(relations_df)} lignes\n")

# Les namespaces définissent les préfixes URI utilisés dans le KB
BASE  = Namespace("http://example.org/mykb/")
ENTITY= Namespace("http://example.org/mykb/entity/")
REL = Namespace("http://example.org/mykb/relation/")
CLASS = Namespace("http://example.org/mykb/class/")
WD = "http://www.wikidata.org/entity/"
WDT= "http://www.wikidata.org/prop/direct/"

# Création du graphe RDF principal
g = Graph()
g.bind("base",BASE)
g.bind("ent", ENTITY)
g.bind("rel", REL)
g.bind("class",CLASS)
g.bind("owl",OWL)
g.bind("rdfs",RDFS)


def clean_text(text):
    if pd.isna(text):
        return ""
    text= str(text).strip()
    text= re.sub(r"\s+", "_", text)
    text= re.sub(r"[^\w\-]", "", text)
    return text

def entity_uri(name):   return URIRef(ENTITY + clean_text(name))
def relation_uri(name): return URIRef(REL    + clean_text(name))
def class_uri(name):    return URIRef(CLASS  + clean_text(name))

# On construit le graphe RDF à partir des entités et relations extraites par spaCy
print("=== Step 1 — Construction du KB privé ===")

known_entities = set()

# Ajout des entités
for _, row in entities_df.iterrows():
    entity= row.get("entity")
    ent_type= row.get("type")
    if not entity:
        continue
    s = entity_uri(entity)
    t = class_uri(ent_type)
    g.add((s, RDF.type, t))
    g.add((s, RDFS.label, Literal(str(entity)))) #label lisible par un humain
    known_entities.add(str(entity).strip())

#Définitions de classes (ontologie)
for ent_type in entities_df["type"].dropna().unique():
    t = class_uri(ent_type)
    g.add((t, RDF.type,  RDFS.Class))
    g.add((t, RDFS.label, Literal(str(ent_type))))
    g.add((t, RDFS.subClassOf, OWL.Thing))

#Ajout des relations
for _, row in relations_df.iterrows():
    subj= row.get("subject")
    pred= row.get("relation")
    obj= row.get("object")
    if not subj or not pred or not obj:
        continue
    s= entity_uri(subj)
    p= relation_uri(pred)
    o= entity_uri(obj) if str(obj).strip() in known_entities else Literal(str(obj))
    g.add((s, p, o))

#Déclarations de propriétés (ontologie)
for pred in relations_df["relation"].dropna().unique():
    p = relation_uri(pred)
    g.add((p, RDF.type,OWL.ObjectProperty))
    g.add((p, RDFS.domain,OWL.Thing))
    g.add((p, RDFS.range, OWL.Thing))
    g.add((p, RDFS.label, Literal(str(pred))))

print(f"  KB privé construit — {len(g)} triplets.\n")


print("=== Step 2 — Alignement des entités (Wikidata) ===")
# Pour chaque entité du KB, on cherche son équivalent dans Wikidata et on ajoute un triplet owl:sameAs pour lier les deux
HEADERS = {
    "User-Agent": "StudentKBBot/1.0 (university project; contact: student@university.edu)"
}

def search_wikidata_entity(label):
    url    = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search":  label,
        "limit": 1
    }
    try:
        resp= requests.get(url, headers=HEADERS, params=params, timeout=10)
        data= resp.json()
        if data.get("search"):
            return data["search"][0]
    except Exception as e:
        print(f"  [Wikidata error] {label}: {e}")
    return None

def compute_confidence(entity, result):
    label_returned= result.get("label", "")
    return round(SequenceMatcher(
        None,
        entity.lower().strip(),
        label_returned.lower().strip()
    ).ratio(), 2)

wikidata_alignments = {}
all_entities= set(entities_df["entity"].dropna().str.strip())

for entity in tqdm(all_entities, desc="  Alignement"):
    private_uri = entity_uri(entity)
    result = search_wikidata_entity(entity)
    time.sleep(0.3)

    if result:
        wikidata_id = result["id"]
        wikidata_uri= URIRef(WD + wikidata_id)
        confidence = compute_confidence(entity, result)

        g.add((private_uri, OWL.sameAs, wikidata_uri))
        g.add((private_uri, URIRef(BASE + "alignmentConfidence"),
               Literal(confidence, datatype=XSD.decimal)))

        wikidata_alignments[entity] = {
            "private_uri": str(private_uri),
            "external_uri":str(wikidata_uri),
            "wikidata_id": wikidata_id,
            "confidence":confidence
        }
    else:
        wikidata_alignments[entity] = {
            "private_uri":str(private_uri),
            "external_uri":None,
            "wikidata_id": None,
            "confidence":0.0
        }
# Sauvegarde de la table d'alignement
df_alignment= pd.DataFrame.from_dict(wikidata_alignments, orient="index")
df_alignment.index.name = "entity"
df_alignment.reset_index(inplace=True)
df_alignment.to_csv("alignment_table.csv", index=False)

aligned_count= len(df_alignment[df_alignment["external_uri"].notna()])
print(f"  {aligned_count}/{len(all_entities)} entités alignées.")
print(f"  Sauvegardé : alignment_table.csv\n")

# On cherche dans Wikidata les propriétés équivalentes à nos prédicats
print("=== Step 3 — Alignement des prédicats (SPARQL) ===")

sparql= SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", HEADERS["User-Agent"])
sparql.setReturnFormat(JSON)

def align_predicate_by_label(pred_label):
    keyword= pred_label.lower().replace("_", " ").strip()
    query= f"""
    SELECT ?property ?propertyLabel WHERE {{
      ?property a wikibase:Property .
      ?property rdfs:label ?propertyLabel .
      FILTER(CONTAINS(LCASE(?propertyLabel), "{keyword}"))
      FILTER(LANG(?propertyLabel) = "en")
    }}
    LIMIT 5
    """
    try:
        sparql.setQuery(query)
        results= sparql.query().convert()
        rows= results["results"]["bindings"]
        return [
            (r["property"]["value"].split("/")[-1], r["propertyLabel"]["value"])
            for r in rows
        ]
    except Exception as e:
        print(f"  [SPARQL error] {pred_label}: {e}")
        return []
# On aligne les 20 prédicats les plus fréquents
predicate_counts = Counter(relations_df["relation"].dropna())
top_predicates= [p for p, _ in predicate_counts.most_common(20)]
predicate_alignments = {}

for pred in tqdm(top_predicates, desc="  Prédicats"):
    candidates= align_predicate_by_label(pred)
    time.sleep(1.0)
    predicate_alignments[pred] = candidates

    if candidates:
        best_id, best_label = candidates[0]
        p_uri = relation_uri(pred)
        g.add((p_uri, OWL.equivalentProperty,
               URIRef(f"{WDT}{best_id}")))

rows_pred= []
for pred, candidates in predicate_alignments.items():
    if candidates:
        best_id, best_label = candidates[0]
        rows_pred.append({
            "private_predicate": pred,
            "wikidata_property": best_id,
            "wikidata_label":    best_label,
            "all_candidates":    str(candidates)
        })
    else:
        rows_pred.append({
            "private_predicate": pred,
            "wikidata_property": None,
            "wikidata_label":    None,
            "all_candidates":    "[]"
        })

df_pred_align = pd.DataFrame(rows_pred)
df_pred_align.to_csv("predicate_alignment.csv", index=False)
print(f"  Sauvegardé : predicate_alignment.csv\n")


print("=== Step 4 — Expansion du KB (SPARQL) ===")
# On n'expand que les entités avec une haute confiance d'alignement 
HIGH_CONF = 0.85
confident_entities = {
    entity: data
    for entity, data in wikidata_alignments.items()
    if data["confidence"] >= HIGH_CONF and data["wikidata_id"] is not None
}
print(f"  {len(confident_entities)} entités à haute confiance sélectionnées.")

def expand_1hop(wikidata_id, limit=300):
    query = f"""
    SELECT ?p ?o WHERE {{
      wd:{wikidata_id} ?p ?o .
      FILTER(!isLiteral(?o))
    }}
    LIMIT {limit}
    """
    try:
        sparql.setQuery(query)
        results= sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"  [1-hop error] {wikidata_id}: {e}")
        return []

def expand_2hop_via_award(limit=3000):
    query = f"""
    SELECT ?person ?award ?p ?o WHERE {{
      ?person wdt:P166 ?award .
      ?award  ?p       ?o .
      FILTER(!isLiteral(?o))
    }}
    LIMIT {limit}
    """
    try:
        sparql.setQuery(query)
        results= sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"  [2-hop error]: {e}")
        return []

# Expansion 1-hop
for entity, data in tqdm(confident_entities.items(), desc="  1-hop expansion"):
    qid = data["wikidata_id"]
    rows= expand_1hop(qid, limit=300)
    time.sleep(1.2)
    for row in rows:
        s_uri= URIRef(WD + qid)
        p_uri= URIRef(row["p"]["value"])
        o_val= row["o"]["value"]
        o_uri= URIRef(o_val) if o_val.startswith("http") else Literal(o_val)
        g.add((s_uri, p_uri, o_uri))

print(f"  Après 1-hop : {len(g)} triplets.")

# Expansion 2-hop
rows_2hop = expand_2hop_via_award(limit=3000)
time.sleep(1.5)
for row in rows_2hop:
    s_uri= URIRef(row["award"]["value"])
    p_uri= URIRef(row["p"]["value"])
    o_val= row["o"]["value"]
    o_uri= URIRef(o_val) if o_val.startswith("http") else Literal(o_val)
    g.add((s_uri, p_uri, o_uri))

print(f"  Après 2-hop : {len(g)} triplets.")

# Suppression des triplets mal formés (sujets ou prédicats sans URI valide)
#on nettoie
print("\n=== Nettoyage ===")
triples_to_remove = [
    (s, p, o) for s, p, o in g
    if not str(s).startswith("http") or not str(p).startswith("http")
]
for triple in triples_to_remove:
    g.remove(triple)
print(f"  {len(triples_to_remove)} triplets mal formés supprimés.")
print(f"  KB final : {len(g)} triplets.\n")


print("=== Export des fichiers finaux ===")

# KB complète
g.serialize("expanded_kb.nt",  format="nt")
g.serialize("expanded_kb.ttl", format="turtle")
print("  Sauvegardé : expanded_kb.nt / expanded_kb.ttl")

# Ontologie seule
g_onto= Graph()
for s, p, o in g:
    if p in (RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range,
             OWL.equivalentProperty, RDFS.label):
        g_onto.add((s, p, o))
g_onto.serialize("ontology.ttl", format="turtle")
print("  Sauvegardé : ontology.ttl")

# Fichier d'alignement (triplets owl:sameAs uniquement)
g_align= Graph()
for s, p, o in g:
    if p == OWL.sameAs or str(p) == str(BASE + "alignmentConfidence"):
        g_align.add((s, p, o))
g_align.serialize("alignment.ttl", format="turtle")
print("  Sauvegardé : alignment.ttl")

# Rapport de statistiques
entities_in_graph= set(s for s, _, _ in g) | set(o for _, _, o in g if isinstance(o, URIRef))
relations_in_graph= set(p for _, p, _ in g)

stats = {
    "total_triples": len(g),
    "total_entities":len(entities_in_graph),
    "total_relations": len(relations_in_graph),
    "aligned_entities":len([v for v in wikidata_alignments.values() if v["external_uri"]]),
    "high_conf_entities":len(confident_entities),
    "top_predicates": dict(predicate_counts.most_common(10))
}

with open("kb_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("  Sauvegardé : kb_stats.json")

print("\n=== Lab 2 terminé ===")
print(f"  Triplets  : {stats['total_triples']}")
print(f"  Entités   : {stats['total_entities']}")
print(f"  Relations : {stats['total_relations']}")
