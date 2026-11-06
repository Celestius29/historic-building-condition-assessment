"""Crack morphology indices (pixels). Copied from notebooks/01_crack_segmentation_deepcrack.ipynb."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects
from skan import Skeleton, summarize


def crack_indices(mask, min_size=20):
    m = remove_small_objects(mask.astype(bool), min_size)
    if m.sum() == 0: return None
    H, W = m.shape
    skel = skeletonize(m)
    w = np.clip(2 * ndi.distance_transform_edt(m)[skel] - 1, 1, None)
    df_ = summarize(Skeleton(skel)); df_.columns = [c.replace("-", "_") for c in df_.columns]
    bl, el = df_["branch_distance"].to_numpy(), df_["euclidean_distance"].to_numpy()
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    n_junc = ndi.label(skel & (nb >= 3), structure=np.ones((3, 3)))[1]
    return dict(length_px=float(bl.sum()), mean_width_px=float(w.mean()),
                tortuosity=float(bl.sum() / max(el.sum(), 1e-6)), n_branches=int(len(df_)),
                n_junctions=int(n_junc), density=float(bl.sum() / (H * W)), area_ratio=float(m.sum() / (H * W)))


def prune_spurs(skel, min_len=10, passes=2):
    skel = skel.copy()
    for _ in range(passes):
        nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
        sk = Skeleton(skel); d = summarize(sk, separator="_")
        short = d[(d["branch_type"] == 1) & (d["branch_distance"] < min_len)].index
        if len(short) == 0: break
        for i in short:
            c = sk.path_coordinates(i).astype(int)
            c = c[nb[c[:, 0], c[:, 1]] < 3]          # keep the junction pixel
            skel[c[:, 0], c[:, 1]] = False
        skel = skeletonize(skel)
    return skel


def branch_stats(mask, min_len):
    m = remove_small_objects(mask.astype(bool), 20)
    if m.sum() == 0: return None
    sk = prune_spurs(skeletonize(m), min_len)
    if sk.sum() < 2: return None
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    nj = ndi.label(sk & (nb >= 3), structure=np.ones((3, 3)))[1]
    return dict(n_branches=len(summarize(Skeleton(sk), separator="_")), n_junctions=nj)
