# Pour lancer: python lab3_part2.py
# Dépendances : pip install pykeen torch scikit-learn matplotlib
# Prérequis : expanded_kb.nt dans le même dossier (généré par lab2.py)


import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#Chargement et nettoyage
print("=== [1] Chargement du KB ===")

# lecture du fichier N-Triples généré par lab2
triples_raw= []
with open("expanded_kb.nt", "r", encoding="utf-8") as f:
    for line in f:
        line= line.strip()
        # Ignorer les commentaires et lignes vides
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip(" .").split(" ", 2)
        if len(parts)== 3:
            s, p, o = parts
            # On garde uniquement les triplets avec des URIs (pas de littéraux)
            if s.startswith("<") and p.startswith("<") and o.startswith("<"):
                triples_raw.append((s, p, o))

print(f"  Triplets chargés (URIs uniquement) : {len(triples_raw)}")

# Suppression des doublons
triples_raw= list(set(triples_raw))
print(f"  Après déduplication : {len(triples_raw)}")

# onfiltre les relations
# On garde seulement les relations avec au moins 100 occurrences pour rester dans la fourchette 50-200 relations demandée
print("\n=== [2] Filtrage des relations ===")

relation_counts= Counter(p for _, p, _ in triples_raw)
print(f"  Relations totales avant filtre : {len(relation_counts)}")

# Seuil dynamique : on ajuste pour avoir entre 50 et 200 relations
seuil= 100
relations_retenues= {r for r, c in relation_counts.items() if c >= seuil}

# Si trop peu de relations, on baisse le seuil
while len(relations_retenues)< 50 and seuil> 10:
    seuil-= 10
    relations_retenues= {r for r, c in relation_counts.items() if c >= seuil}

# Si trop de relations, on monte le seuil
while len(relations_retenues)> 200 and seuil< 10000:
    seuil+= 50
    relations_retenues= {r for r, c in relation_counts.items() if c >= seuil}

print(f"  Seuil appliqué : {seuil} occurrences minimum")
print(f"  Relations retenues : {len(relations_retenues)}")

triples_filtered = [(s, p, o) for s, p, o in triples_raw if p in relations_retenues] # Filtrage des triplets
print(f"  Triplets après filtre : {len(triples_filtered)}")


# On prépare 3 versions du KB : 20k, 50k, et complet
print("\n=== [3] Préparation des sous-ensembles (expérience de taille) ===")

def sample_triples(triples, n):
    """Sous-échantillonne n triplets de façon reproductible."""
    if len(triples)<= n:
        return triples
    random.shuffle(triples)
    return triples[:n]

triples_20k= sample_triples(list(triples_filtered), 20_000)
triples_50k = sample_triples(list(triples_filtered), 50_000)
triples_full= triples_filtered

print(f" 20k: {len(triples_20k)} triplets")
print(f" 50k : {len(triples_50k)} triplets")
print(f" Full: {len(triples_full)} triplets")

# Split train et test
print("\n=== [4] Split Train/Valid/Test ===")

def split_triples(triples, train_ratio=0.8, valid_ratio=0.1):
    """
    Split 80/10/10 en s'assurant que toutes les entités
    apparaissent dans le train (pas de fuite).
    """
    random.shuffle(triples)
    n= len(triples)
    n_train= int(n * train_ratio)
    n_valid= int(n * valid_ratio)

    train= triples[:n_train]
    valid= triples[n_train:n_train + n_valid]
    test= triples[n_train + n_valid:]

    # Vérification : pas d'entité inconnue dans valid/test
    train_entities= set()
    for s, p, o in train:
        train_entities.add(s)
        train_entities.add(o)

    # On retire du valid/test les triplets avec entités inconnues
    valid= [(s, p, o) for s, p, o in valid if s in train_entities and o in train_entities]
    test= [(s, p, o) for s, p, o in test  if s in train_entities and o in train_entities]

    return train, valid, test

def save_splits(train, valid, test, prefix=""):
    """Sauvegarde les splits au format TSV (attendu par PyKEEN)."""
    def write(triples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s, p, o in triples:
                # Nettoyage des < > pour le format TSV
                s= s.strip("<>")
                p= p.strip("<>")
                o= o.strip("<>")
                f.write(f"{s}\t{p}\t{o}\n")

    write(train, f"{prefix}train.txt")
    write(valid, f"{prefix}valid.txt")
    write(test,  f"{prefix}test.txt")
    print(f"  {prefix}train.txt : {len(train)} triplets")
    print(f"  {prefix}valid.txt : {len(valid)} triplets")
    print(f"  {prefix}test.txt  : {len(test)} triplets")

# Split du dataset complet (utilisé pour l'entraînement principal)
train_full, valid_full, test_full= split_triples(triples_full)
save_splits(train_full, valid_full, test_full, prefix="")

# Splits pour l'expérience de taille
train_20k, valid_20k, test_20k= split_triples(triples_20k)
train_50k, valid_50k, test_50k= split_triples(triples_50k)
save_splits(train_20k, valid_20k, test_20k, prefix="20k_")
save_splits(train_50k, valid_50k, test_50k, prefix="50k_")

# ── [5] ENTRAÎNEMENT DES MODÈLES KGE ─────────────────────────
print("\n=== [5] Entraînement des modèles KGE ===")

# Configuration commune aux deux modèles (comparaison équitable)
CONFIG = {
    "num_epochs": 100,
    "embedding_dim": 100,
    "learning_rate": 0.01,
    "batch_size": 512,
    "random_seed": 42,
}

def train_model(model_name, train_path, valid_path, test_path, config):
    print(f"\n  -> Entrainement de {model_name}...")

    result = pipeline(
        model=model_name,
        training=train_path,
        validation=valid_path,
        testing=test_path,
        model_kwargs={"embedding_dim": config["embedding_dim"]},
        training_loop="slcwa",
        training_kwargs={"num_epochs": config["num_epochs"]},
        optimizer="adam",
        optimizer_kwargs={"lr": config["learning_rate"]},
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": 1},
        evaluator_kwargs={"filtered": True},
        random_seed=config["random_seed"],
    )
    return result

# Entraînement sur le dataset complet
result_transe = train_model("TransE",   "train.txt", "valid.txt", "test.txt", CONFIG)
result_distmult= train_model("DistMult", "train.txt", "valid.txt", "test.txt", CONFIG)

# ── [6] ÉVALUATION — LINK PREDICTION ─────────────────────────
print("\n=== [6] Évaluation — Link Prediction ===")

def extract_metrics(result, model_name):
    m = result.metric_results.to_dict()
    both = m.get("both", {}).get("realistic", {})
    return {
        "Modèle": model_name,
        "MRR":  round(both.get("inverse_harmonic_mean_rank", 0), 4),
        "Hits@1":round(both.get("hits_at_1", 0), 4),
        "Hits@3": round(both.get("hits_at_3", 0), 4),
        "Hits@10":round(both.get("hits_at_10", 0), 4),
    }

metrics_transe= extract_metrics(result_transe,   "TransE")
metrics_distmult= extract_metrics(result_distmult, "DistMult")

df_metrics= pd.DataFrame([metrics_transe, metrics_distmult])
df_metrics.to_csv("evaluation_results.csv", index=False)

print("\n  Tableau comparatif des modèles :")
print(df_metrics.to_string(index=False))

# ── [7] EXPÉRIENCE DE TAILLE ──────────────────────────────────
print("\n=== [7] Expérience de taille (TransE) ===")

# On entraîne TransE sur 20k, 50k et full pour observer l'impact
config_light= {**CONFIG, "num_epochs": 50}  # moins d'epochs pour aller vite

result_20k= train_model("TransE", "20k_train.txt",  "20k_valid.txt",  "20k_test.txt",  config_light)
result_50k= train_model("TransE", "50k_train.txt",  "50k_valid.txt",  "50k_test.txt",  config_light)

m_20k= extract_metrics(result_20k,  "TransE 20k")
m_50k= extract_metrics(result_50k,  "TransE 50k")
print("DEBUG m_20k:", m_20k)
print("DEBUG m_50k:", m_50k)
print("DEBUG dict brut:", result_20k.metric_results.to_dict().get("both", {}).get("realistic", {}).get("inverse_harmonic_mean_rank"))
m_full= {**metrics_transe, "Modèle": "TransE Full"}

df_size= pd.DataFrame([m_20k, m_50k, m_full])
df_size.to_csv("size_sensitivity.csv", index=False)

print("\n  Impact de la taille du KB :")
print(df_size.to_string(index=False))

# ── [8] ANALYSE DES EMBEDDINGS ────────────────────────────────
# On utilise le meilleur modèle (celui avec le MRR le plus élevé)
print("\n=== [8] Analyse des embeddings ===")

# Sélection du meilleur modèle
best_result= result_transe if metrics_transe["MRR"] >= metrics_distmult["MRR"] else result_distmult
best_name= "TransE" if metrics_transe["MRR"] >= metrics_distmult["MRR"] else "DistMult"
print(f"  Meilleur modèle : {best_name}")

# Récupération des embeddings d'entités
entity_repr= best_result.model.entity_representations[0]
entity_emb= entity_repr(indices=None).detach().cpu().numpy()
entity_to_id= best_result.training.entity_to_id
id_to_entity= {v: k for k, v in entity_to_id.items()}

print(f"  Embeddings extraits : {entity_emb.shape} (entités × dimensions)")

# ── 8.1 Voisins les plus proches ─────────────────────────────
print("\n  --- 8.1 Voisins les plus proches ---")

# On cherche les voisins pour quelques entités clés
ENTITIES_OF_INTEREST= ["Elon_Musk", "Microsoft", "Google", "France", "Barack_Obama"]
K_NEIGHBORS= 5

nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="cosine").fit(entity_emb)

neighbors_results = {}
for entity_name in ENTITIES_OF_INTEREST:
    # On cherche l'URI correspondante dans le mapping
    matches = [uri for uri in entity_to_id if entity_name.lower() in uri.lower()]
    if not matches:
        print(f"  {entity_name} : non trouvé dans le KB")
        continue

    uri = matches[0]
    idx= entity_to_id[uri]
    emb= entity_emb[idx].reshape(1, -1)
    distances, indices= nbrs.kneighbors(emb)

    # On retire l'entité elle-même (premier voisin = elle-même)
    neighbor_uris=[id_to_entity[i] for i in indices[0][1:]]
    neighbors_results[entity_name]= neighbor_uris

    print(f"\n  Voisins de {entity_name} :")
    for neighbor in neighbor_uris:
        # Affichage raccourci de l'URI
        short = neighbor.split("/")[-1]
        print(f"    → {short}")

with open("nearest_neighbors.json", "w") as f:
    json.dump(neighbors_results, f, indent=2)
print("\n  Sauvegardé : nearest_neighbors.json")

# Clustering t-SNE 
print("\n  --- 8.2 Clustering t-SNE ---")

# Pour t-SNE on prend un échantillon (max 3000 entités pour la lisibilité)
MAX_TSNE= 3000
n_sample= min(MAX_TSNE, len(entity_emb))
sample_ids= random.sample(range(len(entity_emb)), n_sample)

emb_sample = entity_emb[sample_ids]
entity_sample= [id_to_entity[i] for i in sample_ids]

print(f"  Réduction t-SNE sur {n_sample} entités...")
tsne= TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
emb_2d= tsne.fit_transform(emb_sample)

# Coloration par classe ontologique (détectée à partir de l'URI)
def get_class_from_uri(uri):
    """Déduit la classe approximative depuis l'URI Wikidata ou privée."""
    uri_lower= uri.lower()
    if "mykb/entity" in uri_lower:
        return "Private Entity"
    elif "wikidata" in uri_lower:
        return "Wikidata Entity"
    else:
        return "Other"

classes= [get_class_from_uri(e) for e in entity_sample]
unique_classes= list(set(classes))
colors= cm.tab10(np.linspace(0, 1, len(unique_classes)))
class_to_color= dict(zip(unique_classes, colors))

fig, ax= plt.subplots(figsize=(14, 10))
for cls in unique_classes:
    mask= [c == cls for c in classes]
    x= emb_2d[mask, 0]
    y= emb_2d[mask, 1]
    ax.scatter(x, y, c=[class_to_color[cls]], label=cls, alpha=0.5, s=8)

ax.set_title(f"t-SNE des embeddings d'entités ({best_name})", fontsize=14)
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.legend(loc="best", markerscale=3)
plt.tight_layout()
plt.savefig("tsne_clusters.png", dpi=150)
plt.close()
print("  Sauvegardé : tsne_clusters.png")

# ── 8.3 Comportement des relations ───────────────────────────
print("\n  --- 8.3 Comportement des relations ---")

# Récupération des embeddings de relations
rel_repr= best_result.model.relation_representations[0]
rel_emb = rel_repr(indices=None).detach().cpu().numpy()
rel_to_id= best_result.training.relation_to_id
id_to_rel= {v: k for k, v in rel_to_id.items()}

# Détection des relations symétriques (r ≈ -r pour TransE)
# Pour TransE : une relation symétrique devrait avoir ||r|| ≈ 0
print(f"\n  Relations disponibles : {len(rel_to_id)}")
if best_name == "TransE":
    norms= np.linalg.norm(rel_emb, axis=1)
    top_symmetric= np.argsort(norms)[:5]
    print("  Relations potentiellement symétriques (normes faibles) :")
    for idx in top_symmetric:
        rel_uri= id_to_rel[idx]
        short= rel_uri.split("/")[-1]
        print(f"    {short} (norme = {norms[idx]:.4f})")

# exercice 8 RÈGLE SWRL VS EMBEDDING 
print("\n=== [9] Exercice 8 — Comparaison règle SWRL vs embedding ===")

# Règle SWRL : Person(?p) ∧ isFatherOf(?p,?s) ∧ Male(?s) → hasSon(?p,?s)
# On cherche si vector(isFatherOf) + vector(Male) ≈ vector(hasSon)
# dans l'espace d'embedding

print("  Règle SWRL : Person(?p) ∧ isFatherOf(?p,?s) ∧ Male(?s) → hasSon(?p,?s)")
print("  Test embedding : vector(isFatherOf) + vector(Male) ≈ vector(hasSon) ?")

target_rels= ["isFatherOf", "hasSon", "isSonOf"]
found= {}
for rel_name in target_rels:
    matches= [uri for uri in rel_to_id if rel_name.lower() in uri.lower()]
    if matches:
        found[rel_name]= rel_to_id[matches[0]]
        print(f"  Trouvé : {rel_name} → {matches[0]}")
    else:
        print(f"  Non trouvé : {rel_name} (relation absente du KB)")

if len(found)>= 2:
    # Calcul de la similarité cosinus entre les vecteurs de relations
    from numpy.linalg import norm
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

    rel_vecs= {name: rel_emb[idx] for name, idx in found.items()}
    print("\n  Similarités entre relations :")
    rel_names= list(rel_vecs.keys())
    for i in range(len(rel_names)):
        for j in range(i+1, len(rel_names)):
            sim = cosine_sim(rel_vecs[rel_names[i]], rel_vecs[rel_names[j]])
            print(f"    cos({rel_names[i]}, {rel_names[j]}) = {sim:.4f}")


# Recalcul des métriques avec la bonne structure
metrics_transe= extract_metrics(result_transe,   "TransE")
metrics_distmult= extract_metrics(result_distmult, "DistMult")
df_metrics= pd.DataFrame([metrics_transe, metrics_distmult])
print(df_metrics.to_string(index=False))
df_metrics.to_csv("evaluation_results.csv", index=False)



# Export de stats
print("\n=== [10] Export des résultats ===")

rapport = {
    "meilleur_modele": best_name,
    "config": CONFIG,
    "evaluation": {
        "TransE":   metrics_transe,
        "DistMult": metrics_distmult,
    },
    "size_sensitivity": {
        "20k":  m_20k,
        "50k":  m_50k,
        "full": m_full,
    },
    "kb_stats": {
        "triplets_total":    len(triples_filtered),
        "relations_retenues": len(relations_retenues),
        "seuil_filtre":      seuil,
    }
}

with open("lab3_rapport.json", "w") as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)

print("  Sauvegardé : lab3_rapport.json")
print("  Sauvegardé : evaluation_results.csv")
print("  Sauvegardé : size_sensitivity.csv")
print("  Sauvegardé : tsne_clusters.png")
print("  Sauvegardé : nearest_neighbors.json")

print("\n=== Lab 3 Partie 2 terminée ===")
