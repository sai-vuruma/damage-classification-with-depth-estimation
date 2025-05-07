import os
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
import torch
from collections import Counter
import glob

# ---------------------------------------------
# 1. Paths & parameters
# ---------------------------------------------
BASE       = Path(__file__).parent.resolve()
FRAME_DIR  = BASE / 'data/DoriaNET/FRAME'
MASK_DIR   = BASE / 'data/DoriaNET/MASK'
JSON_DIR   = BASE / 'data/DoriaNET/JSON'
DATA_DIR  = BASE / 'dataset'
LABELS_DIR = DATA_DIR / 'labels'
IMAGES_DIR = DATA_DIR / 'images'

CLASSES    = ['damage_0', 'damage_1', 'damage_2', 'damage_3', 'damage_4', 'damage_5']
TRAIN_RATIO= 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
JSON_EXT   = '.json'

# ---------------------------------------------
# 4. Utility: mask → bbox
# ---------------------------------------------
def mask_to_bbox(mask_path):
    mask = Image.open(mask_path).convert('L')
    arr  = np.array(mask)
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2

# ---------------------------------------------
# 5. Write labels & copy images
# ---------------------------------------------
def prepare(split_items, split):
    for jf, img_path in split_items:
        info = json.loads(jf.read_text())
        has_rare_class = any(int(b[3]) == 5 for b in info['Buildings'])  # Check for damage_5
        repeat = 3 if has_rare_class and split == 'train' else 1  # Repeat damage_5 images 3x in training
        for i in range(repeat):
            suffix = f"_rep{i}" if repeat > 1 and i > 0 else ""
            dst_img = IMAGES_DIR / split / (f"{img_path.stem}{suffix}{img_path.suffix}")
            Image.open(img_path).save(dst_img)
            W, H = Image.open(img_path).size
            lines = []
            for b in info['Buildings']:
                mask_path = MASK_DIR / b[2]
                cls = int(b[3])
                bb = mask_to_bbox(mask_path)
                if not bb:
                    continue
                x1, y1, x2, y2 = bb
                xc = ((x1 + x2) / 2) / W
                yc = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl = LABELS_DIR / split / (f"{jf.stem}{suffix}.txt")
            lbl.write_text("\n".join(lines))

# ---------------------------------------------
# 6. Check class balance
# ---------------------------------------------
def check_class_balance():
    for split in ['train', 'val', 'test']:
        labels = glob.glob(str(LABELS_DIR / split / '*.txt'))
        class_counts = Counter()
        for lbl in labels:
            with open(lbl) as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
        print(f"Class counts in {split} set:")
        for cls_idx in range(len(CLASSES)):
            count = class_counts.get(cls_idx, 0)
            print(f"{CLASSES[cls_idx]}: {count}")
        if split == 'train':
            train_counts = class_counts
    return train_counts

def processing():
    # ---------------------------------------------
    # 2. Prepare folders
    # ---------------------------------------------
    for split in ['train', 'val', 'test']:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------
    # 3. Gather JSONs and shuffle
    # ---------------------------------------------
    items = []
    for jf in JSON_DIR.glob(f'*{JSON_EXT}'):
        info = json.loads(jf.read_text())
        frame_name = info.get('Frame_Name')
        if not frame_name:
            continue
        img_path = FRAME_DIR / Path(frame_name).name
        if img_path.exists():
            items.append((jf, img_path))

    random.shuffle(items)
    cut = int(len(items) * TRAIN_RATIO)
    train_items = items[:cut]
    rest_items   = items[cut:]

    cut = int(len(rest_items) * VAL_RATIO / (VAL_RATIO + TEST_RATIO))
    val_items   = rest_items[:cut]
    test_items  = rest_items[cut:]

    prepare(train_items, 'train')
    prepare(val_items,   'val')
    prepare(test_items,   'test')

    train_counts = check_class_balance()

    # ---------------------------------------------
    # 7. Create data.yaml with absolute paths
    # ---------------------------------------------
    config = {
        'train': str((IMAGES_DIR / 'train').resolve()),
        'val':   str((IMAGES_DIR / 'val').resolve()),
        'test':  str((IMAGES_DIR / 'test').resolve()),
        'nc':    len(CLASSES),
        'names': CLASSES
    }

    with open(DATA_DIR / 'data.yaml', 'w') as f:
        yaml.dump(config, f)

    pass