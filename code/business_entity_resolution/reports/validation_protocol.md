# Entity-Disjoint Stratified Validation Protocol

## 1. Protocol Architecture

- **Split Level:** Strict Source 1 entity level (entity-disjoint).
- **Data Leakage Guarantee:** Every candidate pair, matched entity, and feature vector belonging to an S1 entity resides strictly in one partition.
- **Stratification Criteria:** `(country, is_singleton)` to perfectly preserve macro evaluation distributions.
- **Deterministic Seed:** `RandomState(seed=42)`.
- **Artifact Location:** `code/business_entity_resolution/experiments/validation_s1_ids.txt` (441,363 S1 IDs).

## 2. Partition Proportions

| Partition | S1 Entities | Percentage | Singletons | US Count | India Count |
|---|---|---|---|---|---|
| **Train Set (80%)** | 1,765,458 | 80.00% | 98,598 (5.58%) | 1,058,908 (59.98%) | 706,550 (40.02%) |
| **Validation Set (20%)** | 441,363 | 20.00% | 24,649 (5.58%) | 264,725 (59.98%) | 176,638 (40.02%) |
| **Total** | 2,206,821 | 100.00% | 123,247 (5.58%) | 1,323,633 (59.98%) | 883,188 (40.02%) |

## 3. Local Evaluation Rules

- All local tuning, blocking recall thresholds, and classification probability cutoffs must be evaluated exclusively against this 20% holdout.
- Test data must never be used for model selection.
