"""Datasets and client partition schemes.

Partitions (all seeded, all returning per-client index lists):
  P1 pathological : k classes per client, equal sizes
  P2 dirichlet    : label proportions ~ Dir(alpha) per class across clients
  P3 powerlaw     : IID labels, power-law sample counts
  P4 compound     : Dirichlet labels x power-law sizes

`synthetic` provides an offline stand-in (Gaussian class blobs) so the full
pipeline runs without downloads (CI / smoke tests).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class SyntheticImageDataset(Dataset):
    """Gaussian class-blob images; linearly separable enough to learn quickly.

    `means_seed` fixes the class means (SHARED between train and test splits);
    `seed` varies only the labels and additive noise.
    """

    def __init__(self, n: int, num_classes: int, shape: Tuple[int, int, int],
                 seed: int, means_seed: int = 0) -> None:
        c, h, w = shape
        means = np.random.default_rng(means_seed).normal(
            0, 1.0, size=(num_classes, c, h, w)).astype(np.float32)
        rng = np.random.default_rng(seed)
        self.targets = torch.from_numpy(rng.integers(0, num_classes, size=n))
        data = means[self.targets.numpy()] + rng.normal(
            0, 0.7, size=(n, c, h, w)
        ).astype(np.float32)
        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], int(self.targets[i])


def load_dataset(name: str, root: str, seed: int):
    """Return (train_set, test_set, num_classes, input_shape)."""
    name = name.lower()
    if name == "synthetic":
        shape, ncls = (1, 28, 28), 10
        return (
            SyntheticImageDataset(6000, ncls, shape, seed),
            SyntheticImageDataset(1000, ncls, shape, seed + 1),
            ncls,
            shape,
        )
    from torchvision import datasets as tvd
    from torchvision import transforms as T

    if name == "mnist":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tr = tvd.MNIST(root, train=True, download=True, transform=tf)
        te = tvd.MNIST(root, train=False, download=True, transform=tf)
        return tr, te, 10, (1, 28, 28)
    if name in ("fmnist", "fashionmnist", "fashion-mnist"):
        tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        tr = tvd.FashionMNIST(root, train=True, download=True, transform=tf)
        te = tvd.FashionMNIST(root, train=False, download=True, transform=tf)
        return tr, te, 10, (1, 28, 28)
    if name == "cifar10":
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        tr = tvd.CIFAR10(root, train=True, download=True, transform=tf)
        te = tvd.CIFAR10(root, train=False, download=True, transform=tf)
        return tr, te, 10, (3, 32, 32)
    raise ValueError(f"unknown dataset {name}")


def _labels_of(ds) -> np.ndarray:
    t = ds.targets
    return t.numpy() if torch.is_tensor(t) else np.asarray(t)


# -----------------------------------------------------------------------------
# Partition schemes
# -----------------------------------------------------------------------------

def partition_pathological(labels: np.ndarray, num_clients: int, k: int,
                           num_classes: int, rng: np.random.Generator
                           ) -> List[np.ndarray]:
    """k classes per client via class-shards (McMahan-style)."""
    shards_per_class = max(1, num_clients * k // num_classes)
    class_shards: List[np.ndarray] = []
    shard_class: List[int] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for chunk in np.array_split(idx, shards_per_class):
            if len(chunk):
                class_shards.append(chunk)
                shard_class.append(c)
    order = rng.permutation(len(class_shards))
    parts: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    got: List[set] = [set() for _ in range(num_clients)]
    leftovers = []
    for si in order:  # first pass: distinct classes per client where possible
        placed = False
        for ci in rng.permutation(num_clients):
            if len(parts[ci]) < k and shard_class[si] not in got[ci]:
                parts[ci].append(class_shards[si])
                got[ci].add(shard_class[si])
                placed = True
                break
        if not placed:
            leftovers.append(si)
    for si in leftovers:  # second pass: fill remaining slots
        for ci in np.argsort([len(p) for p in parts]):
            if len(parts[ci]) < k:
                parts[ci].append(class_shards[si])
                break
    return [np.concatenate(p) if p else np.array([], dtype=int) for p in parts]


def partition_dirichlet(labels: np.ndarray, num_clients: int, alpha: float,
                        num_classes: int, rng: np.random.Generator,
                        min_size: int = 10) -> List[np.ndarray]:
    n = len(labels)
    while True:
        parts: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet([alpha] * num_clients)
            # cap clients already above average to keep sizes sane
            props = np.array([
                p * (len(pi) < n / num_clients) for p, pi in zip(props, parts)
            ])
            s = props.sum()
            props = props / s if s > 0 else np.full(num_clients, 1 / num_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for ci, chunk in enumerate(np.split(idx, cuts)):
                parts[ci].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]


def partition_powerlaw(labels: np.ndarray, num_clients: int, exponent: float,
                       rng: np.random.Generator, min_size: int = 10
                       ) -> List[np.ndarray]:
    n = len(labels)
    raw = np.array([(i + 1) ** (-exponent) for i in range(num_clients)])
    sizes = np.maximum((raw / raw.sum() * n).astype(int), min_size)
    idx = rng.permutation(n)
    out, pos = [], 0
    for s in sizes:
        out.append(idx[pos: pos + s])
        pos = min(pos + s, n)
    return out


def partition_compound(labels: np.ndarray, num_clients: int, alpha: float,
                       exponent: float, num_classes: int,
                       rng: np.random.Generator) -> List[np.ndarray]:
    """Dirichlet label skew, then subsample per client to power-law sizes."""
    parts = partition_dirichlet(labels, num_clients, alpha, num_classes, rng)
    raw = np.array([(i + 1) ** (-exponent) for i in range(num_clients)])
    rng.shuffle(raw)
    keep = raw / raw.max()
    out = []
    for p, f in zip(parts, keep):
        m = max(10, int(len(p) * f))
        out.append(rng.choice(p, size=min(m, len(p)), replace=False))
    return out


def make_partitions(cfg, train_set, num_classes: int, rng: np.random.Generator
                    ) -> List[np.ndarray]:
    labels = _labels_of(train_set)
    p = cfg.partition
    if p == "iid":
        idx = rng.permutation(len(labels))
        return [a for a in np.array_split(idx, cfg.num_clients)]
    if p == "pathological":
        return partition_pathological(labels, cfg.num_clients,
                                      cfg.classes_per_client, num_classes, rng)
    if p == "dirichlet":
        return partition_dirichlet(labels, cfg.num_clients, cfg.dirichlet_alpha,
                                   num_classes, rng)
    if p == "powerlaw":
        return partition_powerlaw(labels, cfg.num_clients, cfg.powerlaw_exponent,
                                  rng)
    if p == "compound":
        return partition_compound(labels, cfg.num_clients, cfg.dirichlet_alpha,
                                  cfg.powerlaw_exponent, num_classes, rng)
    raise ValueError(f"unknown partition {p}")


def client_histograms(train_set, parts: List[np.ndarray], num_classes: int
                      ) -> np.ndarray:
    """True per-client label histograms (LOCAL knowledge; never released raw)."""
    labels = _labels_of(train_set)
    H = np.zeros((len(parts), num_classes))
    for i, p in enumerate(parts):
        if len(p):
            H[i] = np.bincount(labels[p], minlength=num_classes)
    return H


def build_loaders(train_set, test_set, parts: List[np.ndarray], batch_size: int,
                  val_fraction: float, rng: np.random.Generator
                  ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """Client train loaders + server-side public validation split + test loader.

    The validation split is carved from the *test* set (server-side public data)
    so ESCA's accuracy weighting consumes no client privacy budget.
    """
    loaders = [
        DataLoader(Subset(train_set, p.tolist()), batch_size=batch_size,
                   shuffle=True, drop_last=False)
        for p in parts
    ]
    n_test = len(test_set)
    perm = rng.permutation(n_test)
    n_val = max(1, int(val_fraction * n_test))
    val = DataLoader(Subset(test_set, perm[:n_val].tolist()), batch_size=256)
    test = DataLoader(Subset(test_set, perm[n_val:].tolist()), batch_size=256)
    return loaders, val, test
