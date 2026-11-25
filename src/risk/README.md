# src/risk : modèle d'estimation du risque

Pour chaque image vidéo (5 par seconde), le modèle prédit si le conducteur va **freiner fort dans les 2 secondes**.
Il utilise trois sources d'information :

- **les objets détectés par YOLOv8s** : usagers vulnérables (VRU), véhicules, objets dans la trajectoire, et leur rapprochement ;
- **la carte d'attention BDDA** : la part de l'attention posée sur les VRU, sur les véhicules et sur la route devant ;
- **le GPS** : la vitesse, l'accélération passée et la tendance de la vitesse.

Le **label** vient de la décélération GPS, donc aucune annotation manuelle n'est nécessaire. Les clips BDDA ont été
découpés environ 6.5 s avant un freinage, ce qui veut dire que chaque clip doit en contenir un.

> La carte du regard BDDA est la moyenne de 45 observateurs en laboratoire. Elle montre où un conducteur attentif
> devrait regarder. Ce n'est pas le regard du conducteur qui a filmé la vidéo.

Le code ne modifie jamais `data/BDDA/` ni `data/processed/`. Les résultats vont dans `data/risk/` et `reports/risk/`,
deux dossiers ignorés par Git.

## Prérequis

- Le dossier `data/BDDA/` avec ses trois splits (`training`, `validation`, `test`). Chaque split contient
  `camera_videos`, `gazemap_videos` et `gps_jsons`.
- Le fichier `reports/video_dataset/dataset_manifest.csv`, déjà présent dans le dépôt.
- Le fichier `weights/final/yolov8s_v2_best.pt`, déjà présent dans le dépôt.
- `pip install -r requirements.txt`. Ce fichier contient déjà tout ce qu'il faut : ultralytics, scikit-learn, opencv, pandas et matplotlib.

## Commandes à lancer, depuis la racine du dépôt

| étape | commande | ce qu'elle produit | durée |
|---|---|---|---|
| 1. Labels | `python src/risk/braking_labels.py` | `data/risk/braking_events.csv` et `reports/risk/braking_labels_report.md` | moins d'une minute (GPS uniquement) |
| 2. Prétraitement complet | `python src/risk/run_full_preprocessing.py` | les images de `data/processed/` pour les 1 161 séquences | quelques heures, environ 13 Go |
| 3. Détection | `python src/risk/detect_objects.py` | `data/risk/detections.csv` | environ 30 à 40 min sur une RTX 2050, peut reprendre après une interruption |
| 4. Features | `python src/risk/build_features.py` | `data/risk/frame_features.csv` | quelques minutes |
| 5. Modèle | `python src/risk/train_risk_model.py` | `reports/risk/risk_model_report.md`, les fichiers `metrics_*.json` et les figures | une à quelques minutes |

**Tester d'abord sur le pilote.** Tu peux sauter l'étape 2 et lancer les étapes 3 à 5 directement sur les 9 séquences
du pilote, déjà présentes dans `data/processed/`. Le but est seulement de vérifier que tout s'exécute : avec 3 séquences
par split, les chiffres ne veulent rien dire.

**Vérifier l'étape 1 avant de continuer.** Dans `braking_labels_report.md`, la majorité des débuts de freinage doivent
se situer entre 4.5 et 8.5 s. Si ce n'est pas le cas, ou si trop de clips n'ont aucun événement, essaie un autre seuil :
`python src/risk/braking_labels.py --decel 1.5`. Il faut ensuite relancer les étapes 4 et 5.

**Le split test est protégé.** Il n'est évalué qu'une seule fois, à la fin, avec `python src/risk/train_risk_model.py --final-test`.

## Ce qui est comparé

| modèle | rôle |
|---|---|
| `rule baseline` | un score calculé par une formule fixe, sans apprentissage (`rule_baseline.py`) |
| `logistic` | une régression logistique |
| `boosting` | du gradient boosting (HistGradientBoosting de scikit-learn) |
| `time-only (ref.)` | une référence qui ne voit que le temps écoulé dans le clip. Comme chaque clip est coupé environ 6.5 s avant le freinage, le temps suffit en partie à deviner la réponse. Ce modèle mesure ce raccourci : ce n'est pas un candidat. |

Les métriques sont :
- l'AP (PR-AUC), qui est la métrique principale, et la ROC-AUC ;
- l'**AUC à temps égal** : elle compare des images prises au même moment du clip, donc le raccourci du temps ne joue pas ;
- les métriques par clip : alarme à temps, alarme trop tôt, ou alarme manquée, et l'avance médiane ;
- des intervalles de confiance à 95 % par bootstrap sur les séquences ;
- une ablation qui mesure ce qu'apporte chaque groupe de features (objets, mouvement, attention, cinématique).

## Les fichiers

| fichier | rôle |
|---|---|
| `risk_config.py` | tous les paramètres : seuils, horizon, chemins |
| `gps_utils.py` | lecture du GPS et calcul de la cinématique *causale*, c'est-à-dire sans jamais lire le futur |
| `records.py` | lecture des `aligned_records.csv`. Si les chemins absolus enregistrés n'existent plus, ils sont reconstruits. |
| `braking_labels.py` | étape 1 |
| `run_full_preprocessing.py` | étape 2 : lance `src/features/video_preprocessing.py` sur toutes les séquences, sans modifier sa configuration |
| `detect_objects.py` | étape 3 |
| `build_features.py` | étape 4 |
| `rule_baseline.py` | la formule du score à base de règles |
| `train_risk_model.py` | étape 5 |

Le pipeline a été vérifié de bout en bout sur des données synthétiques. La détection a aussi été testée avec le vrai
fichier `yolov8s_v2_best.pt`.
