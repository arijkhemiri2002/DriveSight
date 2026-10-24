import zipfile, os
from collections import Counter

# ---------- À ADAPTER ----------
IMG_ZIP    = r"C:\Users\arijk\Downloads\bdd100k_images_10k.zip"   # votre zip d'images 10K
LABELS_ZIP = r"C:\Users\arijk\Downloads\bdd100k_labels.zip"
OUT_DIR    = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\dataset"
MAX_PER_SPLIT = {"train": 2500, "val": 500}   # limite pour économiser l'espace ; None = tout
# --------------------------------

def stem(p): return os.path.splitext(os.path.basename(p))[0]
def split_of(p): return p.replace("\\", "/").split("/")[-2]

# 1) images du zip, avec leur split (train/val/test)
with zipfile.ZipFile(IMG_ZIP) as zi:
    img_by_name = {stem(n): n for n in zi.namelist() if n.lower().endswith(".jpg")}
print("Images 10K :", len(img_by_name))

# 2) labels du zip, indexés par nom
with zipfile.ZipFile(LABELS_ZIP) as zl:
    lab_by_name = {stem(n): n for n in zl.namelist() if n.endswith(".json")}
    print("Labels :", len(lab_by_name))

    common = sorted(set(img_by_name) & set(lab_by_name))
    print("Paires image + label :", len(common))

    # 3) extraction par split, avec limite
    counts = Counter()
    with zipfile.ZipFile(IMG_ZIP) as zi:
        for name in common:
            sp = split_of(img_by_name[name])
            if sp == "test":                       # pas de labels publics utiles
                continue
            lim = (MAX_PER_SPLIT or {}).get(sp)
            if lim is not None and counts[sp] >= lim:
                continue
            os.makedirs(os.path.join(OUT_DIR, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(OUT_DIR, "labels_json", sp), exist_ok=True)
            with open(os.path.join(OUT_DIR, "images", sp, name + ".jpg"), "wb") as f:
                f.write(zi.read(img_by_name[name]))
            with open(os.path.join(OUT_DIR, "labels_json", sp, name + ".json"), "wb") as f:
                f.write(zl.read(lab_by_name[name]))
            counts[sp] += 1

print("Extrait :", dict(counts))
print("->", OUT_DIR)