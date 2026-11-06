"""Data pairing, datasets and split saving. Logic copied from notebooks/01_crack_segmentation_deepcrack.ipynb."""
import glob, json, os, random
import numpy as np
import torch
from PIL import Image

ROOT = "DeepCrack/dataset"
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def pairs(split):
    labs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{ROOT}/{split}_lab/*")}
    return [(i, labs[os.path.splitext(os.path.basename(i))[0]]) for i in sorted(glob.glob(f"{ROOT}/{split}_img/*"))]


class DS(torch.utils.data.Dataset):
    def __init__(self, items, aug=False): self.items, self.aug = items, aug
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ip, lp = self.items[i]
        img = np.asarray(Image.open(ip).convert("RGB").resize((544, 384)), dtype=np.float32) / 255
        m = (np.asarray(Image.open(lp).convert("L").resize((544, 384), Image.NEAREST)) > 127).astype(np.float32)
        if self.aug:
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
        img = ((img - MEAN) / STD).transpose(2, 0, 1)
        return torch.tensor(img.copy(), dtype=torch.float32), torch.tensor(m.copy())[None]


class DS2(torch.utils.data.Dataset):
    def __init__(self, items, train=False): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ip, lp = self.items[i]
        img = np.asarray(Image.open(ip).convert("RGB").resize((544, 384)), dtype=np.float32) / 255
        m = (np.asarray(Image.open(lp).convert("L").resize((544, 384), Image.NEAREST)) > 127).astype(np.float32)
        if self.train:
            x0 = random.randint(0, 544 - 384); img, m = img[:, x0:x0 + 384], m[:, x0:x0 + 384]
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
            img = np.clip(img * random.uniform(.8, 1.2) + random.uniform(-.1, .1), 0, 1)
        img = ((img - MEAN) / STD).transpose(2, 0, 1)
        return torch.tensor(img.copy(), dtype=torch.float32), torch.tensor(m.copy())[None]


def make_split():
    """Notebook split code wrapped in a function (seed 0, 60 of the 300 official train images for validation)."""
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    train_all, test = pairs("train"), pairs("test")
    random.shuffle(train_all)
    n_val = int(0.2 * len(train_all)); val, train = train_all[:n_val], train_all[n_val:]
    return train, val, test


def save_split(train, val, test, path="results/metrics/split_seed0.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"train": [os.path.basename(a) for a, _ in train], "val": [os.path.basename(a) for a, _ in val],
               "test": [os.path.basename(a) for a, _ in test]}, open(path, "w"))
