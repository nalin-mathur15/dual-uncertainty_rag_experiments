"""Data pipeline for constructing 500k–1M+ Dual-Uncertainty General RAG states."""
from .schema import (
    ACTION_ANSWER,
    ACTION_ASK,
    ACTION_SEARCH,
    ACTION_ABSTAIN,
    ACTION_NAMES,
    QUADRANT_NAMES,
    RAGStateExample,
)
from .hf_ingestor import stream_raw_qa_records
from .perturbation_engine import DualUncertaintyPerturbationEngine
from .dataset import ShardedRAGDataset, create_sharded_mmap_split
