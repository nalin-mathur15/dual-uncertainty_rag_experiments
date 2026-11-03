"""Memory-mapped sharded dataset storage for 500k–1M+ General RAG states.

Uses NumPy `.npy` memory-mapped arrays inside each split directory so that
training over 1,000,000+ examples requires minimal RAM and supports fast
multi-worker PyTorch DataLoaders on SLURM GPU clusters.
"""
import os
import json
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .schema import (
    DEFAULT_EMBED_DIM,
    QUERY_STAT_DIM,
    PASSAGE_STAT_DIM,
    QUESTION_STAT_DIM,
    REWRITE_STAT_DIM,
    RawQARecord,
)
from .perturbation_engine import DualUncertaintyPerturbationEngine


ARRAY_SPECS = {
    "query_vec": ((DEFAULT_EMBED_DIM + QUERY_STAT_DIM,), np.float32),
    "passage_vecs": ((6, DEFAULT_EMBED_DIM + PASSAGE_STAT_DIM), np.float32),
    "passage_mask": ((6,), np.bool_),
    "question_vecs": ((8, DEFAULT_EMBED_DIM + QUESTION_STAT_DIM), np.float32),
    "rewrite_vecs": ((4, DEFAULT_EMBED_DIM + REWRITE_STAT_DIM), np.float32),
    "flat_summary": ((164,), np.float32),
    "prompt_certainty_label": ((), np.float32),
    "evidence_certainty_label": ((), np.float32),
    "quadrant_label": ((), np.int64),
    "optimal_macro_action": ((), np.int64),
    "current_answer_regret": ((), np.float32),
    "is_safe_to_answer": ((), np.float32),
    "is_unsupported_evidence": ((), np.float32),
    "macro_action_values": ((4,), np.float32),
    "question_voi": ((8,), np.float32),
    "question_ig": ((8,), np.float32),
    "rewrite_voi": ((4,), np.float32),
    "best_question_idx": ((), np.int64),
    "best_rewrite_idx": ((), np.int64),
    "latent_intent_target": ((8,), np.float32),
}


def create_sharded_mmap_split(
    split_dir: str,
    num_states: int,
    raw_records: List[RawQARecord],
    engine: DualUncertaintyPerturbationEngine,
    rng: np.random.Generator,
    split_mode: str = "in_distribution",
) -> None:
    """Generates `num_states` examples directly into disk-backed memory-mapped `.npy` files."""
    os.makedirs(split_dir, exist_ok=True)
    mmaps = {}
    for name, (shape, dtype) in ARRAY_SPECS.items():
        full_shape = (num_states,) + shape
        path = os.path.join(split_dir, f"{name}.npy")
        mmaps[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=full_shape)

    n_raw = len(raw_records)
    for i in tqdm(range(num_states), desc=f"Building {os.path.basename(split_dir)} ({num_states} states)"):
        rec = raw_records[i % n_raw]
        ex = engine.generate_state(rec, rng, quadrant=None, split_mode=split_mode)
        for name in ARRAY_SPECS:
            mmaps[name][i] = getattr(ex, name)

    for mm in mmaps.values():
        mm.flush()

    meta = {"num_states": num_states, "split_mode": split_mode}
    with open(os.path.join(split_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


class ShardedRAGDataset(Dataset):
    """Zero-copy memory-mapped PyTorch Dataset for hundreds of thousands or millions of RAG states."""

    def __init__(self, split_dir: str, max_samples: int = None):
        self.split_dir = split_dir
        with open(os.path.join(split_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        total = int(meta["num_states"])
        self.length = min(total, max_samples) if max_samples is not None else total
        self.mmaps = {}
        for name in ARRAY_SPECS:
            path = os.path.join(split_dir, f"{name}.npy")
            self.mmaps[name] = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {}
        for name, arr in self.mmaps.items():
            val = np.array(arr[idx], copy=True)
            if val.dtype == np.bool_:
                out[name] = torch.from_numpy(val).bool()
            elif val.dtype == np.int64:
                out[name] = torch.from_numpy(val).long()
            else:
                out[name] = torch.from_numpy(val).float()
        return out
