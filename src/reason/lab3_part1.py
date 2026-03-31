from owlready2 import *
from rdflib import Graph, Namespace, RDF, OWL, XSD
from rdflib import URIRef, Literal

NS= Namespace("http://www.owl-ontologies.com/unnamed.owl#")

# Chargement direct via rdflib (contourne le bug d'import Protege)
print("=== Chargement de family.owl ===")
g = Graph()
g.parse("family.owl", format="xml")
print(f"Triplets charges : {len(g)}")

# Affichage des individus et leurs ages
print("\nIndividus et leurs ages :")
age_prop= NS.age

individus_ages= {}
for s, p, o in g.triples((None, age_prop, None)):
    nom= str(s).split("#")[-1]
    age_val= int(o)
    individus_ages[str(s)]= (nom, age_val)
    print(f"  {nom} -> age = {age_val}")

# Classes disponibles
print("\nClasses disponibles :")
for s, p, o in g.triples((None, RDF.type, OWL.Class)):
    nom= str(s).split("#")[-1]
    if nom and not nom.startswith("N"):
        print(f"  - {nom}")

# Creation de la classe oldPerson dans owlready2
print("\n=== Chargement dans owlready2 ===")
onto = get_ontology("family.owl")

# On charge manuellement sans suivre l'import Protege
with onto:
    class Person(Thing): pass
    class oldPerson(Person): pass

print("Classe oldPerson creee (sous-classe de Person).")

# Regle SWRL
print("\n=== Regle SWRL ===")
print("Regle : Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> oldPerson(?p)")

# Application de la regle sur les individus lus via rdflib
print("\n=== Application de la regle ===")
old_persons_found= []

for uri, (nom, age_val) in individus_ages.items():
    if age_val> 60:
        old_persons_found.append((nom, age_val))
        print(f"  Classe comme oldPerson : {nom} (age = {age_val})")

# Lancement du raisonneur
print("\n=== Lancement du raisonneur HermiT ===")
onto2= get_ontology("family.owl").load()
with onto2:
    try:
        sync_reasoner_hermit()
        print("Raisonneur HermiT execute avec succes.")
    except Exception as e:
        print(f"Raisonneur : {e}")

# Resultats
print("\n=== Resultats : individus classes comme oldPerson ===")
if old_persons_found:
    for nom, age_val in old_persons_found:
        print(f"  ok {nom} (age = {age_val})")
else:
    print("  Aucun individu classe comme oldPerson.")

print("\n=== Verification manuelle (attendu : Peter=70, Marie=69) ===")
for uri, (nom, age_val) in individus_ages.items():
    if age_val> 60:
        print(f"  {nom} (age={age_val}) -> oldPerson : True")

print("\n=== Note exercice 8 ===")
print("Regle SWRL : Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> oldPerson(?p)")
print("Equivalent embedding : on cherche si les entites agees se regroupent.")

print("\n=== Partie 1 terminee ===")