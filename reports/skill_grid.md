# Mission 9: Grille d'évaluation des compétences

**Projet**: Traitement Big Data sur le Cloud  
**Date**: 2026-01-11  
**Statut**: En cours

---

## COMPÉTENCE 1: Sélectionner les outils du Cloud pour le Big Data

> Sélectionner les outils du Cloud permettant de traiter et stocker les données d'un projet Big Data conforme aux normes RGPD en vigueur afin de concevoir une application de qualité supportant le traitement de données massives.

| CE | Critère | Statut | Implémentation | Preuve |
|---|---|---|---|---|
| **CE1** | Identifier les différentes briques d'architecture nécessaires pour la mise en place d'un environnement Big Data | ✅ FAIT | Architecture documentée: S3 (stockage), EMR (calcul distribué), JupyterHub (notebook), Spark (traitement) | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 1 |
| **CE2** | Identifier les outils du cloud permettant de mettre en place l'environnement Big Data conforme aux normes RGPD | ✅ FAIT | AWS région eu-west-1/eu-west-3 (Europe), données stockées et traitées sur territoire européen | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 4 + Section 14 |

**Résumé Module 1**: ✅ **100% COMPLET** (2/2 critères)

---

## COMPÉTENCE 2: Prétraiter et modéliser des données Big Data

> Prétraiter, analyser et modéliser des données (en veillant à leur conformité RGPD) dans un environnement Big Data et en utilisant les outils du Cloud afin de concevoir une application sécurisée de qualité supportant le traitement de données massives.

| CE | Critère | Statut | Implémentation | Preuve |
|---|---|---|---|---|
| **CE1** | Charger les fichiers de départ et ceux après transformation dans un espace de stockage cloud conforme à la réglementation RGPD | ✅ FAIT | Images chargées depuis S3 (`s3://bucket/Test`), résultats sauvegardés sur S3 (`s3://bucket/Results`) en région EU | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 4 + Section 12 |
| **CE2** | Exécuter les scripts en utilisant des machines dans le cloud | ✅ FAIT | Exécution sur cluster EMR (1 Master + 2 Workers m5.xlarge), Spark distribue les tâches | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 9 |
| **CE3** | Réaliser un script qui permet d'écrire les sorties du programme directement dans l'espace de stockage cloud | ✅ FAIT | `features_df.write.parquet(PATH_Result)` écrit directement sur S3, export CSV également | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 12 |

**Résumé Module 2**: ✅ **100% COMPLET** (3/3 critères)

---

## COMPÉTENCE 3: Réaliser des calculs distribués sur des données massives

> Réaliser des calculs distribués sur des données massives en utilisant les outils adaptés et en prenant en compte le RGPD afin de permettre la mise en œuvre d'applications à l'échelle.

| CE | Critère | Statut | Implémentation | Preuve |
|---|---|---|---|---|
| **CE1** | Identifier les traitements critiques lors d'un passage à l'échelle en termes de volume de données | ✅ FAIT | Broadcast des poids du modèle (`sc.broadcast()`), repartition pour parallélisation, Pandas UDF pour batch processing | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 7-8 |
| **CE2** | Veiller à ce que l'exploitation des données soit conforme au RGPD (serveurs sur territoire européen) | ✅ FAIT | Bucket S3 en eu-west-1/eu-west-3, cluster EMR dans même région, aucune donnée hors UE | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Section 4 |
| **CE3** | Développer les scripts s'appuyant sur Spark | ✅ FAIT | PySpark DataFrame, `spark.read.format("binaryFile")`, Pandas UDF, PCA Spark ML | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) Sections 5-12 |
| **CE4** | S'assurer que toute la chaîne de traitement est exécutée dans le cloud | ✅ FAIT | Notebook exécuté sur JupyterHub EMR, lecture S3 → traitement Spark → écriture S3 | [notebooks/mission9.ipynb](../notebooks/mission9.ipynb) |

**Résumé Module 3**: ✅ **100% COMPLET** (4/4 critères)

---

## Récapitulatif Global

| Compétence | Critères | Statut |
|---|---|---|
| C1: Outils Cloud Big Data | 2/2 | ✅ 100% |
| C2: Prétraitement Big Data | 3/3 | ✅ 100% |
| C3: Calculs distribués | 4/4 | ✅ 100% |
| **TOTAL** | **9/9** | **✅ 100%** |

---

## Livrables

1. ✅ **Notebook PySpark** - [notebooks/mission9.ipynb](../notebooks/mission9.ipynb)
   - Preprocessing des images
   - Extraction features MobileNetV2 avec broadcast weights
   - Réduction dimension PCA

2. ✅ **Données sur le Cloud**
   - Images source: `s3://bucket/Test/` (22688 images)
   - Features 1280-dim: `s3://bucket/Results/` (Parquet)
   - Features PCA 50-dim: `s3://bucket/Results_PCA/` (Parquet)
   - Export CSV: `s3://bucket/Results_CSV/`

3. ✅ **Support de présentation** - À créer
   - Architecture cloud (S3, EMR, JupyterHub)
   - Démarche Big Data
   - Étapes PySpark
