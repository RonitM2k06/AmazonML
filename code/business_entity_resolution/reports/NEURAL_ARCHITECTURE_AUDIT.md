# Neural Architecture Audit — Amazon ML Challenge 2026

Generated: 2026-09-26

---

## 1. Current Champion Pipeline Audit

### Champion (EXP_007)
| Metric | Value |
|---|---|
| Macro F0.5 | **0.951463** |
| Macro Precision | 0.976974 |
| Macro Recall | 0.898266 |
| Singleton Accuracy | 96.92% |
| Exact Set Accuracy | 71.35% |
| Architecture | XGB/LGB/CAT ensemble (45/45/10) |
| Feature Dim | 56 hand-crafted features |
| Decision Strategy | `transitive_cross_source_consistency_bridge` |
| Blocking | Union E (95.78% recall @ 204 cands/S1) |

### Reusable Components

| Component | File | Reuse in Neural Pipeline |
|---|---|---|
| Data loader | `scripts/run_phase5_experiment.py:load_train_and_val_samples()` | ✅ Used directly |
| Ground truth reader | Same file | ✅ Used directly |
| Evaluation (Macro F0.5) | `src/evaluation.py:Evaluator` | ✅ Used directly |
| Entity decision engine | `src/entity_decision.py` | ✅ Reusable with neural scores |
| Union E blocking | `src/blocking.py:build_target_index()` | ✅ Used in EXP_N002 union |
| Feature extraction | `src/pair_features.py` | ✅ Usable for hybrid EXP_N008 |
| XGB/LGB/CAT models | `models/xgboost_matcher.pkl` | ✅ Champion baseline preserved |

---

## 2. Neural Architecture Plan

### Why Neural Retrieval?

The current 14.41% blocking miss rate is driven by:
- **69.8%** of misses: script/transliteration/name representation problems
  - Example: `Fortune Management` vs `फॉर्च्यून मैनेजमेंट`
  - Phonetic blocking partially addressed this (EXP_008, +2.5% recall)
  - Neural embeddings from multilingual MiniLM can bridge this gap via joint multilingual space

- **~4.2%** of true matches: no valid blocking key in any strategy
  - These can ONLY be recovered by embedding-based semantic retrieval

### Architecture Overview

```
S1 entity
    ↓
[Bi-Encoder] paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
    ↓
[FAISS IndexFlatIP] Country-partitioned ANN search
    ↓ top-200 candidates
[Optional: Cross-Encoder reranking] mmarco-mMiniLMv2-L6-H384-v1
    ↓ top-50 reranked
[Entity Decision] threshold on cosine / ce_score
    ↓
Predicted match set
```

### Model Choices

| Model | Type | Params | Dim | Languages | VRAM (fp32, batch=256) |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | Bi-encoder | 118M | 384 | 50+ incl. Hindi/French | ~640MB |
| `mmarco-mMiniLMv2-L6-H384-v1` | Cross-encoder | 22M | — | 100+ | ~128MB/batch-128 |

Both confirmed downloadable and working on this machine.

---

## 3. Experiment Plan (EXP_N001–EXP_N010)

| Exp | Description | Goal |
|---|---|---|
| **EXP_N001** | Dual encoder ANN only, threshold sweep | Baseline neural recall and F0.5 |
| **EXP_N002** | ANN ∪ Union E, cosine gate for UE candidates | Test if union improves recall |
| **EXP_N003** | Cross-encoder reranking of top-50 ANN | Test if CE reranking beats simple cosine |
| **EXP_N004** | Dual encoder + contrastive loss fine-tuning | Can training improve embeddings for this domain? |
| **EXP_N005** | Dual encoder + transliteration augmentation | Better representation of Indic names |
| **EXP_N006** | Cross-encoder standalone (no ANN) | Upper bound: how good is CE alone? |
| **EXP_N007** | Cross-encoder + hard negatives from ANN | Fine-tuned CE on domain hard negatives |
| **EXP_N008** | Hybrid: 56 XGB features + neural cosine score | Neural feature added to existing pipeline |
| **EXP_N009** | Hybrid + S2↔S3 cross-source graph bridge | Graph consistency with neural scores |
| **EXP_N010** | Full: ANN → CE reranking → graph → decision | Complete neural pipeline |

---

## 4. Memory and Compute Budget

| Resource | Available | Required (Neural POC) |
|---|---|---|
| GPU VRAM | 6 GB (RTX 4050) | ~1.5 GB (bi-encoder encoding) |
| RAM | 16 GB | ~4–6 GB (embeddings + FAISS) |
| Disk | N/A | ~1.2 GB (S1 + target embeddings at 25k) |
| CPU Cores | 28 logical | Used for data loading |

**Key constraint**: 25,000 S1 × 384 float32 = 38 MB (trivial).
Target embeddings (matches + 60k background): ~100k × 384 × 4B = 154 MB (trivial).
Full-dataset encoding (all S2+S3 ~5M records): ~7.5 GB → requires chunked encoding.

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Neural F0.5 < champion on POC | High (cosine is noisy) | Use hybrid (EXP_N008) to combine |
| Cross-encoder model not downloadable | Low (confirmed working) | Fall back to bi-encoder |
| VRAM OOM during encoding | Low (batch=256, 640MB/batch) | Reduce batch size or use CPU |
| Country mismatch in FAISS index | Low (country partition verified) | Assert all query countries exist |
| Neural recall < Union E recall | Medium | Union of both (EXP_N002) |

---

## 6. Files Created

| File | Purpose |
|---|---|
| `src/neural_encoder.py` | Multilingual MiniLM bi-encoder with lazy loading and caching |
| `src/neural_retrieval.py` | FAISS country-partitioned ANN retriever with Recall@K evaluation |
| `src/cross_encoder.py` | Cross-encoder reranker with batched pairwise scoring |
| `scripts/run_neural_poc.py` | Master runner: EXP_N001 + N002 + N003 |
| `reports/NEURAL_ARCHITECTURE_AUDIT.md` | This document |
