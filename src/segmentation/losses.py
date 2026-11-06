"""clDice loss. Copied from notebooks/01_crack_segmentation_deepcrack.ipynb."""
import torch
import torch.nn.functional as F


def soft_skel(x, k=10):
    er = lambda t: torch.minimum(-F.max_pool2d(-t, (3, 1), 1, (1, 0)), -F.max_pool2d(-t, (1, 3), 1, (0, 1)))
    op = lambda t: F.max_pool2d(er(t), 3, 1, 1)
    s = F.relu(x - op(x))
    for _ in range(k):
        x = er(x); d = F.relu(x - op(x)); s = s + F.relu(d - s * d)
    return s


def cldice(logits, y, k=10, eps=1e-6):
    p = torch.sigmoid(logits); sp, st = soft_skel(p, k), soft_skel(y, k)
    tp = ((sp * y).sum() + eps) / (sp.sum() + eps); ts = ((st * p).sum() + eps) / (st.sum() + eps)
    return 1 - 2 * tp * ts / (tp + ts)
