import json, glob, os, random, shutil
from collections import Counter, defaultdict

# ---------- À ADAPTER ----------
BASE   = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\dataset"
IMG_IN = os.path.join(BASE, "images", "train")
LAB_IN = os.path.join(BASE, "labels_json", "train")
OUT    = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\yolo"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MOVE   = False        # False = copie (garde dataset intact) ; True = déplace (économise de la place)
SEED   = 42
W, H   = 1280, 720
# --------------------------------

# 9 classes : "train" retirée. Noms BDD100K -> indice YOLO
CLASSES = ["pedestrian", "rider", "car", "truck", "bus",
           "motorcycle", "bicycle", "traffic light", "traffic sign"]
BDD2ID = {"person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
          "motor": 5, "bike": 6, "traffic light": 7, "traffic sign": 8}
RARE_TAGS = {"night", "snowy", "rainy", "foggy"}   # contextes à répartir aussi entre les splits

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

# 1) Images disponibles (dataset, et ancien dossier yolo si une conversion précédente a déjà déplacé des images)
sources = {}
for pattern in (os.path.join(IMG_IN, "*.jpg"),
                os.path.join(OUT, "images", "*", "*.jpg")):
    for p in glob.glob(pattern):
        sources.setdefault(stem(p), p)

# 2) Lecture des labels : boîtes YOLO + "empreinte" de chaque image pour la stratification
data, skipped = {}, 0
for lp in glob.glob(os.path.join(LAB_IN, "*.json")):
    name = stem(lp)
    if name not in sources:
        continue
    with open(lp) as f:
        d = json.load(f)
    attrs = d.get("attributes", {})
    lines, feats = [], Counter()
    for fr in d.get("frames", []):
        for o in fr.get("objects", []):
            if "box2d" not in o or o["category"] not in BDD2ID:
                continue
            b = o["box2d"]
            x1, x2 = max(0, min(W, b["x1"])), max(0, min(W, b["x2"]))
            y1, y2 = max(0, min(H, b["y1"])), max(0, min(H, b["y2"]))
            if x2 - x1 < 1 or y2 - y1 < 1:
                skipped += 1
                continue
            cid = BDD2ID[o["category"]]
            lines.append(f"{cid} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            feats[CLASSES[cid]] += 1
    for tag in (attrs.get("timeofday"), attrs.get("weather")):
        if tag in RARE_TAGS:
            feats["tag:" + tag] += 1
    data[name] = {"lines": lines, "feats": feats}

names = sorted(data)
print(f"Images avec image + label : {len(names)} | boîtes dégénérées ignorées : {skipped}")

# 3) Split stratifié glouton : déficit RELATIF, classes rares pondérées plus fort
feats = {n: Counter(data[n]["feats"]) for n in data}
for n in feats:
    feats[n]["_img"] = 1                      # compte aussi les effectifs d'images

totals = Counter()
for n in feats:
    totals.update(feats[n])

def rarity(n):                                # plus c'est grand, plus l'image est "rare"
    return sum(c / totals[k] for k, c in feats[n].items() if k != "_img")

names = sorted(data)
random.seed(SEED)
random.shuffle(names)
names.sort(key=rarity, reverse=True)          # les images rares sont placées en premier

assigned = {s: Counter() for s in RATIOS}
n_split = Counter()
split_of = {}
for n in names:
    best, best_score = None, None
    for s, r in RATIOS.items():
        score = 0.0
        for k, c in feats[n].items():
            target = r * totals[k]
            score += (c / totals[k]) * (target - assigned[s][k]) / target
        if best_score is None or score > best_score:
            best, best_score = s, score
    split_of[n] = best
    assigned[best].update(feats[n])
    n_split[best] += 1


# 4) Écriture : dossiers propres, étiquettes YOLO, images
shutil.rmtree(os.path.join(OUT, "labels"), ignore_errors=True)

for s in RATIOS:
    os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)

for n in names:
    s = split_of[n]

    # Écrire les annotations YOLO
    label_path = os.path.join(OUT, "labels", s, n + ".txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(data[n]["lines"]))

    # Supprimer les anciennes copies issues du split précédent
    for s2 in RATIOS:
        old = os.path.join(OUT, "images", s2, n + ".jpg")

        if (
            s2 != s
            and os.path.exists(old)
            and not sources[n].startswith(OUT)
        ):
            os.remove(old)

    # Copier ou déplacer l'image
    src = sources[n]
    dst = os.path.join(OUT, "images", s, n + ".jpg")

    if os.path.abspath(src) == os.path.abspath(dst):
        continue

    if MOVE or os.path.abspath(src).startswith(os.path.abspath(OUT)):
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


# 5) data.yaml
with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {OUT.replace(os.sep, '/')}\n")
    f.write("train: images/train\nval: images/val\ntest: images/test\n")
    f.write(f"nc: {len(CLASSES)}\nnames:\n")
    for i, c in enumerate(CLASSES):
        f.write(f"  {i}: {c}\n")

# 6) Liste des splits (utile pour relier chaque image à son fichier info)
with open(os.path.join(OUT, "split.csv"), "w") as f:
    f.write("video,split\n")
    for n in sorted(split_of):
        f.write(f"{n},{split_of[n]}\n")

# 7) Rapport
print("\nImages par split :", dict(n_split))
print(f"\n{'classe':<16}" + "".join(f"{s:>9}" for s in RATIOS))
for k in CLASSES + ["tag:night", "tag:snowy", "tag:rainy"]:
    print(f"{k:<16}" + "".join(f"{assigned[s][k]:>9}" for s in RATIOS))
print("\nTerminé ->", OUT)