import os, glob, random
import cv2

OUT = r"C:\Users\arijk\Desktop\DriveSight\DriveSight\data\yolo"
W, H = 1280, 720
CLASSES = ["pedestrian", "rider", "car", "truck", "bus",
           "motorcycle", "bicycle", "traffic light", "traffic sign"]
stem = lambda p: os.path.splitext(os.path.basename(p))[0]

# 1) cohérence des fichiers
seen = {}
for s in ("train", "val", "test"):
    imgs = {stem(p) for p in glob.glob(os.path.join(OUT, "images", s, "*.jpg"))}
    labs = {stem(p) for p in glob.glob(os.path.join(OUT, "labels", s, "*.txt"))}
    print(f"{s:<6} images {len(imgs):>5} | labels {len(labs):>5} | "
          f"sans label {len(imgs - labs)} | sans image {len(labs - imgs)}")
    for n in imgs:
        if n in seen:
            print("  DOUBLON entre", seen[n], "et", s, ":", n)
        seen[n] = s

# 2) indices de classe et coordonnées
bad = 0
for p in glob.glob(os.path.join(OUT, "labels", "*", "*.txt")):
    for line in open(p).read().splitlines():
        c, *v = line.split()
        if not (0 <= int(c) < len(CLASSES)) or any(not 0 <= float(x) <= 1 for x in v):
            bad += 1
print("Lignes invalides :", bad)

# 3) contrôle visuel : une image par split
random.seed(1)
for s in ("train", "val", "test"):
    p = random.choice(glob.glob(os.path.join(OUT, "images", s, "*.jpg")))
    img = cv2.imread(p)
    lab = os.path.join(OUT, "labels", s, stem(p) + ".txt")
    for line in open(lab).read().splitlines():
        c, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc)*W, float(yc)*H, float(w)*W, float(h)*H
        p1, p2 = (int(xc-w/2), int(yc-h/2)), (int(xc+w/2), int(yc+h/2))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(img, CLASSES[int(c)], (p1[0], p1[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(OUT, f"check_{s}.jpg"), img)
print("Ouvrez check_train.jpg, check_val.jpg, check_test.jpg dans", OUT)