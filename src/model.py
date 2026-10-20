"""
Model code: frozen CLIP image encoder + trainable projection head.

Pipeline:  image -> CLIP ViT-B/32 (frozen) -> 512-d embedding
                 -> ProjectionHead (trained with triplet loss) -> 512-d embedding
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMB_DIM = 512


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------
LOCAL_CLIP_WEIGHTS = (Path(__file__).resolve().parent.parent
                      / "models" / "clip" / "open_clip_pytorch_model.bin")


def load_clip(device="cpu"):
    """Load the CLIP image encoder and its preprocessing transform.

    Uses manually downloaded weights in models/clip/ when present,
    otherwise downloads them from the Hugging Face Hub.
    """
    import open_clip
    if LOCAL_CLIP_WEIGHTS.exists():
        # The OpenAI weights use QuickGELU, so the matching config is required
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained=str(LOCAL_CLIP_WEIGHTS))
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def embed_pil(images, model, preprocess, device="cpu", batch_size=64):
    """Embed a list of PIL images -> (N, 512) L2-normalised float32 array."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(im.convert("RGB"))
                             for im in images[i:i + batch_size]]).to(device)
        f = model.encode_image(batch).float()
        feats.append(F.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def embed_paths(paths, model, preprocess, device="cpu", batch_size=64, log_every=20):
    """Embed image files in batches (images are opened lazily to save memory)."""
    feats = []
    n_batches = (len(paths) + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, len(paths), batch_size), start=1):
        images = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        feats.append(embed_pil(images, model, preprocess, device, batch_size))
        if b % log_every == 0 or b == n_batches:
            print(f"  batch {b}/{n_batches}")
    return np.concatenate(feats)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """Residual MLP on top of CLIP embeddings.

    output = normalize(x + MLP(x)).  The last layer starts at zero, so before
    training the head returns the CLIP embedding unchanged (= the baseline).
    Training can therefore only move away from the baseline when it helps.
    """

    def __init__(self, dim=EMB_DIM, hidden=512):
        super().__init__()
        self.dim, self.hidden = dim, hidden
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.mlp(x), dim=-1)


def save_head(head, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "dim": head.dim,
                "hidden": head.hidden, "extra": extra or {}}, path)


def load_head(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    head = ProjectionHead(ckpt["dim"], ckpt["hidden"])
    head.load_state_dict(ckpt["state_dict"])
    return head.eval().to(device)


@torch.no_grad()
def apply_head(head, emb, device="cpu", batch_size=2048):
    """Run the projection head over a NumPy array of CLIP embeddings."""
    head.eval()
    out = []
    for i in range(0, len(emb), batch_size):
        x = torch.as_tensor(emb[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(head(x).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def batch_hard_triplet_loss(emb, labels, margin=0.2):
    """Batch-hard triplet loss on L2-normalised embeddings (cosine distance).

    For each anchor: hardest positive = same label, largest distance;
                     hardest negative = different label, smallest distance.
    Returns (loss, fraction of triplets that are still active).
    """
    dist = 1.0 - emb @ emb.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    hardest_pos = dist.masked_fill(~(same & ~eye), float("-inf")).max(dim=1).values
    hardest_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
    losses = F.relu(hardest_pos - hardest_neg + margin)
    return losses.mean(), (losses > 0).float().mean()


def pk_batches(labels, P=32, K=4, rng=None):
    """Yield index batches with P different parents x K images each.

    Every batch is guaranteed to contain positives and negatives for every
    anchor, which batch-hard mining needs.
    """
    rng = rng or np.random.default_rng()
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    parents = np.array(list(groups))
    rng.shuffle(parents)
    for start in range(0, len(parents) - P + 1, P):
        idx = []
        for p in parents[start:start + P]:
            members = groups[int(p)]
            idx.extend(rng.choice(members, K, replace=len(members) < K))
        yield np.array(idx)
