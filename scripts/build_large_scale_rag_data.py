"""Step 1: Build Large-Scale (500k to 1M+) General-Purpose Dual-Uncertainty RAG Benchmark.

Ingests real public QA & RAG datasets (AmbigQA, HotpotQA, MS MARCO, and SciFact via
HuggingFace `datasets`, with automatic offline fallback) and applies the 2x2x2
Counterfactual Perturbation Engine to generate memory-mapped `.npy` splits:
  - train (default: 500,000 states; set --scale 2.0 for 1,000,000 states)
  - val (default: 25,000 states)
  - test_id (default: 50,000 in-distribution states)
  - test_domain_shift (default: 20,000 SciFact/BioMed domain-shift states)
  - test_adversarial_evid (default: 20,000 adversarial contradiction/retrieval-failure states)
"""
import argparse
import os
import yaml
import numpy as np

from src.data_pipeline.hf_ingestor import stream_raw_qa_records
from src.data_pipeline.perturbation_engine import DualUncertaintyPerturbationEngine
from src.data_pipeline.dataset import create_sharded_mmap_split


def main():
    parser = argparse.ArgumentParser(description="Build 500k-1M+ General RAG Dual-Uncertainty Benchmark.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale multiplier for dataset size (1.0 = 590k total states, 2.0 = 1.18M states, 0.02 = fast sanity check)",
    )
    parser.add_argument("--offline", action="store_true", help="Skip HuggingFace Hub download and use fast procedural generator")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    env_cfg = cfg["rag_environment"]
    engine = DualUncertaintyPerturbationEngine(
        num_passages=env_cfg["num_retrieved_passages"],
        num_questions=env_cfg["num_candidate_questions"],
        num_rewrites=env_cfg["num_candidate_rewrites"],
        question_cost_lambda=env_cfg["question_cost_lambda"],
        search_cost_gamma=env_cfg["search_cost_gamma"],
        hallucination_penalty=env_cfg["hallucination_penalty"],
        abstain_regret=env_cfg["abstain_regret"],
    )

    data_dir = cfg["storage"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    n_train = max(500, int(cfg["splits"]["train_states"] * args.scale))
    n_val = max(200, int(cfg["splits"]["val_states"] * args.scale))
    n_test_id = max(200, int(cfg["splits"]["test_id_states"] * args.scale))
    n_test_dom = max(200, int(cfg["splits"]["test_domain_shift_states"] * args.scale))
    n_test_adv = max(200, int(cfg["splits"]["test_adversarial_evid_states"] * args.scale))

    print(f"[DataBuilder] Ingesting raw QA records (use_hf_hub={not args.offline})...")
    raw_id_records = stream_raw_qa_records(
        split_mode="in_distribution",
        max_raw_records=min(50000, max(2000, n_train // 4)),
        use_hf_hub=not args.offline,
        seed=cfg["seed"],
    )
    raw_ood_records = stream_raw_qa_records(
        split_mode="domain_shift",
        max_raw_records=min(15000, max(1000, n_test_dom)),
        use_hf_hub=not args.offline,
        seed=cfg["seed"] + 7,
    )

    # Split raw_id_records into disjoint train / val / test pools so there is zero query leakage!
    n_raw = len(raw_id_records)
    t_cut = int(n_raw * 0.80)
    v_cut = int(n_raw * 0.90)
    train_pool = raw_id_records[:t_cut]
    val_pool = raw_id_records[t_cut:v_cut]
    test_pool = raw_id_records[v_cut:]

    create_sharded_mmap_split(os.path.join(data_dir, "train"), n_train, train_pool, engine, rng, "in_distribution")
    create_sharded_mmap_split(os.path.join(data_dir, "val"), n_val, val_pool, engine, rng, "in_distribution")
    create_sharded_mmap_split(os.path.join(data_dir, "test_id"), n_test_id, test_pool, engine, rng, "in_distribution")
    create_sharded_mmap_split(
        os.path.join(data_dir, "test_domain_shift"), n_test_dom, raw_ood_records, engine, rng, "domain_shift"
    )
    create_sharded_mmap_split(
        os.path.join(data_dir, "test_adversarial_evid"), n_test_adv, test_pool, engine, rng, "adversarial_evidence"
    )

    total_generated = n_train + n_val + n_test_id + n_test_dom + n_test_adv
    print(f"\n[DataBuilder] Successfully generated {total_generated:,} General RAG states in {data_dir}/!")


if __name__ == "__main__":
    main()
