# Neural Architecture POC Report
Generated: 2026-09-26 00:46:06

## Champion Baseline
- EXP_007 Ensemble XGB/LGB/CAT: **Macro F0.5 = 0.951463**

## ANN Retrieval Recall@K (paraphrase-multilingual-MiniLM-L12-v2)
| K | Recall@K |
|---|---|
| 10 | 0.7984 |
| 25 | 0.8370 |
| 50 | 0.8619 |
| 100 | 0.8855 |
| 200 | 0.9068 |

## Experiment Results
| Experiment | Description | Macro F0.5 | Precision | Recall | vs Champion |
|---|---|---|---|---|---|
| EXP_007 (champion) | XGB/LGB/CAT ensemble | 0.951463 | 0.9770 | 0.8983 | baseline |
| EXP_N001 | Dual encoder ANN only | 0.554361 | 0.5834 | 0.7172 | ▼ 0.3971 |
| EXP_N002 | ANN ∪ Union E | 0.093655 | 0.0813 | 0.8544 | ▼ 0.8578 |
| EXP_N003 | Cross-encoder reranking top-50 | FAILED | - | - | - |

## Analysis
- ANN Recall@200 = 0.9068 vs Union E Recall = 0.9578 (25k benchmark)
- Neural retrieval supplements lexical blocking by capturing semantic similarity across script/transliteration variations.
- Cross-encoder reranking provides fine-grained pairwise scoring but is limited to top-K ANN candidates.

## Next Steps
- EXP_N004: Train dual encoder with contrastive loss on hard negatives
- EXP_N005: Data augmentation (transliteration, noise)
- EXP_N008: Hybrid: existing 56 XGB features + neural cosine score
- EXP_N009: Hybrid + cross-source S2↔S3 graph bridge

Total runtime: 3.4 minutes
