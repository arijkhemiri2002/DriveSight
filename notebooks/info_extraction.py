import zipfile, os, glob, json

# ---------- À ADAPTER ----------
INFO_ZIP = r"C:\Users\arijk\Downloads\bdd100k_info.zip"
LAB_DIR  = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\dataset\labels_json\train"
OUT_INFO = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\dataset\info\train"
# --------------------------------
os.makedirs(OUT_INFO, exist_ok=True)

stem = lambda p: os.path.splitext(os.path.basename(p))[0]

wanted = {stem(p) for p in glob.glob(os.path.join(LAB_DIR, "*.json"))}
print("Noms du dataset :", len(wanted))

with zipfile.ZipFile(INFO_ZIP) as z:
    info_by_name = {stem(n): n for n in z.namelist() if n.endswith(".json")}
    print("Fichiers info dans le zip :", len(info_by_name))
    print("Exemples :", list(info_by_name.items())[:3])

    matched = sorted(wanted & info_by_name.keys())
    missing = sorted(wanted - info_by_name.keys())
    print(f"Correspondances : {len(matched)} / {len(wanted)}")

    for name in matched:
        with open(os.path.join(OUT_INFO, name + ".json"), "wb") as f:
            f.write(z.read(info_by_name[name]))

# vérification : clés d'un fichier et lien rideID <-> nom
if matched:
    with open(os.path.join(OUT_INFO, matched[0] + ".json")) as f:
        d = json.load(f)
    print("\nFichier :", matched[0])
    print("Clés :", list(d.keys()))
    print("rideID :", d.get("rideID"))

if missing:
    print("Exemples sans info :", missing[:5])