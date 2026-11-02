"""Move the tasks of a source dataset whose image has no golden on another host into a side dataset (kept under
datasets/, so select_swesmith.py still counts them as used); the side dataset is screened on node-94 later.
usage: partition.py SRC_DATASET GOLDEN_KEYS_FILE SIDE_DATASET   (keys: sha1(image)[:16], the golden/.golden/*.cid names)"""
import hashlib, os, shutil, sys, tomllib

D = "/home/user/harbor-udocker/datasets"
src, side = f"{D}/{sys.argv[1]}", f"{D}/{sys.argv[3]}"
keys = set(open(sys.argv[2]).read().split())
os.makedirs(side, exist_ok=True)
moved = 0
for t in sorted(os.listdir(src)):
    image = tomllib.load(open(f"{src}/{t}/task.toml", "rb"))["environment"]["docker_image"]
    if hashlib.sha1(image.encode()).hexdigest()[:16] not in keys:
        shutil.move(f"{src}/{t}", f"{side}/{t}"); moved += 1
print(f"{sys.argv[1]}: {len(os.listdir(src))} stay, {moved} moved to {sys.argv[3]} (now {len(os.listdir(side))})")
