# LAB 4 — RAG avec RDF/SPARQL et LLM local (Ollama)
# Pour lancer :
#   python lab4_rag_sparql.py
# Dépendances: pip install rdflib requests
# Prérequis : expanded_kb.ttl dans le même dossier


import re
import time
import requests
from typing import List, Tuple
from rdflib import Graph

# Configuration 
TTL_FILE   = "expanded_kb.ttl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2:1b"

# Requêtes SPARQL prédéfinies pour la démo
# Ces requêtes sont adaptées à la structure exacte du KB
DEMO_QUESTIONS = [
    {
        "question": "What type of entity is Microsoft?",
        "sparql": """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ent:  <http://example.org/mykb/entity/>
SELECT ?type WHERE {
  ent:Microsoft rdf:type ?type .
}
"""
    },
    {
        "question": "What are the relations of Google in the knowledge base?",
        "sparql": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ent:  <http://example.org/mykb/entity/>
SELECT ?relation ?object WHERE {
  ent:Google ?relation ?object .
  FILTER(?relation != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
} LIMIT 10
"""
    },
    {
        "question": "What entities are of type PERSON?",
        "sparql": """
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX class: <http://example.org/mykb/class/>
SELECT ?entity ?label WHERE {
  ?entity rdf:type class:PERSON .
  ?entity rdfs:label ?label .
} LIMIT 10
"""
    },
    {
        "question": "What is Elon Musk linked to in the knowledge base?",
        "sparql": """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ent:  <http://example.org/mykb/entity/>
SELECT ?relation ?object WHERE {
  ent:Elon_Musk ?relation ?object .
} LIMIT 20
"""
    },
    {
        "question": "What organizations are in the knowledge base?",
        "sparql": """
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX class: <http://example.org/mykb/class/>
SELECT ?entity ?label WHERE {
  ?entity rdf:type class:ORG .
  ?entity rdfs:label ?label .
} LIMIT 10
"""
    },
]

# Appel au LLM local via Ollama
def ask_llm(prompt: str) -> str:
    """Envoie un prompt au LLM local et retourne la reponse."""
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200, "num_ctx": 1024}
    }
    try:
        t0 = time.time()
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        print(f"  [LLM] {time.time() - t0:.1f}s")
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print("[ERREUR] Ollama n'est pas lance.")
        exit(1)
    except requests.exceptions.ReadTimeout:
        print("[ERREUR] Timeout.")
        return ""

# Chargement du graphe RDF 
def load_graph(ttl_path: str) -> Graph:
    print(f"Chargement de {ttl_path}...")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"  {len(g)} triplets charges.")
    return g

# Execution SPARQL 
def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    """Execute une requete SPARQL sur le graphe."""
    res   = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows

#  Baseline: reponse sans RAG 
def answer_no_rag(question: str) -> str:
    """Reponse directe du LLM sans acces au KB (baseline)."""
    prompt = f"Answer this question as best as you can:\n\n{question}"
    return ask_llm(prompt)

# Formulation de la reponse finale
def formulate_answer(question: str, rows: list, vars_: list) -> str:
    """
    Etape finale du RAG :
    Le LLM recoit les resultats SPARQL et formule une reponse
    en langage naturel ancree dans les donnees du KB.
    """
    if not rows:
        return "Aucun resultat trouve dans le KB."
    results_text = "\n".join([" | ".join(r) for r in rows])
    prompt = (
        "List ALL the entities linked to the subject based on the facts below.\n"
        "List every entity, one per line.\n\n"
        f"Question: {question}\n\n"
        f"Facts from knowledge base:\n{results_text}\n\n"
        "Answer:"
    )
    return ask_llm(prompt)

# Affichage des resultats
def print_result(question: str, sparql: str, vars_: list, rows: list, answer: str):
    print("\n[Requete SPARQL utilisee]")
    print(sparql.strip())

    if not rows:
        print("\n[Aucun resultat retourne par le KB]")
        return

    print(f"\n[Resultats bruts du KB] ({len(rows)} lignes)")
    print(" | ".join(vars_))
    print("-" * 60)
    for r in rows[:10]:
        print(" | ".join(r))
    if len(rows) > 10:
        print(f"... ({len(rows)} resultats au total)")

    print("\n[Reponse finale (RAG)]")
    print(answer)

#  Demo
if __name__ == "__main__":
    g = load_graph(TTL_FILE)

    print("\n" + "=" * 60)
    print("  RAG SPARQL - KB Wikipedia + LLM local (Ollama)")
    print(f"  Modele : {MODEL}")
    print("=" * 60)

    # Mode demo : on parcourt les 5 questions predefinies
    print("\nMode DEMO — 5 questions predefinies\n")

    for i, item in enumerate(DEMO_QUESTIONS):
        question = item["question"]
        sparql   = item["sparql"]

        print(f"\n{'='*60}")
        print(f"Question {i+1} : {question}")
        print(f"{'='*60}")

        # Baseline
        print("\n--- Baseline (LLM sans KB) ---")
        baseline = answer_no_rag(question)
        print(baseline)

        # RAG : execution SPARQL + reponse LLM
        print("\n--- RAG SPARQL (LLM + KB RDF) ---")
        try:
            vars_, rows = run_sparql(g, sparql)
            answer = formulate_answer(question, rows, vars_) if rows else "Aucun resultat."
            print_result(question, sparql, vars_, rows, answer)
        except Exception as e:
            print(f"[Erreur SPARQL] : {e}")

    # Mode interactif apres la demo
    print("\n" + "=" * 60)
    print("Mode INTERACTIF — pose tes propres questions")
    print("Tape 'quit' pour quitter.")
    print("=" * 60)

    while True:
        question = input("\nQuestion : ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
        if not question:
            continue

        print("\n--- Baseline (LLM sans KB) ---")
        print(answer_no_rag(question))

        # Pour le mode interactif on essaie de generer le SPARQL
        print("\n--- RAG SPARQL (LLM + KB RDF) ---")
        print("[Note] En mode interactif, la generation SPARQL automatique")
        print("       peut echouer avec un petit modele comme llama3.2:1b.")

        # Requete generique par label
        sparql_generic = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?p ?o WHERE {{
  ?s rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{question.split()[-1]}")))
  ?s ?p ?o .
}} LIMIT 10
"""
        try:
            vars_, rows = run_sparql(g, sparql_generic)
            answer = formulate_answer(question, rows, vars_) if rows else "Aucun resultat trouve."
            print_result(question, sparql_generic, vars_, rows, answer)
        except Exception as e:
            print(f"[Erreur] : {e}")