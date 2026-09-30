# Amazon ML Challenge 2026: Comprehensive Dataset Profile

**Date Generated:** 2026-09-25 20:07:41
**Problem:** Business Entity Resolution

---

## 1. Executive Summary

The Amazon ML Challenge 2026 dataset consists of business records across 3 independent data sources:
- **Source 1 (S1):** Deduplicated reference ground-truth entities.
- **Source 2 (S2) & Source 3 (S3):** Noisy external sources containing spelling variants, transliterations, abbreviations, and informal address references.
- **Ground Truth:** Maps each Source 1 entity to 0, 1, or multiple matching entities in Source 2 and Source 3.

- **Total Dataset Size:** ~2403.80 MB (2.35 GB)
- **Total Source Records:** 24,229,173 across train and test
- **Total Training S1 Entities:** 2,206,821
- **Total Test S1 Entities:** 1,732,544
- **Ground Truth Matches:** 7,638,365 link pairs across S2 and S3
- **Singletons (No Match):** 123,247 (5.58%)

---

## 2. File Inventory & Specifications

| File | Path | Size (MB) | Row Count | Columns | Malformed Rows |
|---|---|---|---|---|---|
| `train_source1.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\train\train_source1.tsv` | 200.34 | 2,206,821 | entity_id, business_name, business_address, country | 0 |
| `train_source2.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\train\train_source2.tsv` | 466.63 | 5,034,616 | entity_id, business_name, business_address, country | 0 |
| `train_source3.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\train\train_source3.tsv` | 480.37 | 5,285,603 | entity_id, business_name, business_address, country | 0 |
| `test_source1.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\test\test_source1.tsv` | 166.91 | 1,732,544 | entity_id, business_name, business_address, country | 0 |
| `test_source2.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\test\test_source2.tsv` | 485.86 | 4,887,273 | entity_id, business_name, business_address, country | 0 |
| `test_source3.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\test\test_source3.tsv` | 482.56 | 5,082,316 | entity_id, business_name, business_address, country | 0 |
| `train_ground_truth.tsv` | `C:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\dataset\train\train_ground_truth.tsv` | 121.13 | 2,206,821 | source1_entity_id, matched_entity_ids | 0 |

---

## 3. Data Integrity & Completeness

| File | Unique IDs | Duplicate IDs | Missing Names | Missing Addrs | Missing Country |
|---|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `train_source2.tsv` | 5,034,616 | 0 | 0 (0.0%) | 168,967 (3.3561%) | 0 (0.0%) |
| `train_source3.tsv` | 5,285,603 | 0 | 0 (0.0%) | 175,916 (3.3282%) | 0 (0.0%) |
| `test_source1.tsv` | 1,732,544 | 0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `test_source2.tsv` | 4,887,273 | 0 | 0 (0.0%) | 129,408 (2.6479%) | 0 (0.0%) |
| `test_source3.tsv` | 5,082,316 | 0 | 0 (0.0%) | 136,098 (2.6779%) | 0 (0.0%) |

---

## 4. Country Distributions

Notice that the **training** set strictly covers `US` and `India`, whereas the **test** set introduces `France` as an open-set category!

| File | US Count (%) | India Count (%) | France Count (%) | Total Rows |
|---|---|---|---|---|
| `train_source1.tsv` | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.0%) | 2,206,821 |
| `train_source2.tsv` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.0%) | 5,034,616 |
| `train_source3.tsv` | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.0%) | 5,285,603 |
| `test_source1.tsv` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 1,732,544 |
| `test_source2.tsv` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 4,887,273 |
| `test_source3.tsv` | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.4%) | 5,082,316 |

---

## 5. String Length Distributions

### Business Name Length (Characters)

| File | Min | Max | Mean ± Std | Median | P25 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 3 | 105 | 24.03 ± 7.74 | 24 | 18 | 30 | 34 | 37 | 42 |
| `train_source2.tsv` | 2 | 104 | 25.1 ± 8.89 | 25 | 19 | 31 | 37 | 40 | 48 |
| `train_source3.tsv` | 2 | 123 | 25.2 ± 9.49 | 25 | 18 | 31 | 37 | 42 | 50 |
| `test_source1.tsv` | 3 | 92 | 23.84 ± 7.67 | 24 | 18 | 29 | 34 | 36 | 42 |
| `test_source2.tsv` | 2 | 102 | 25.7 ± 9.12 | 25 | 19 | 32 | 38 | 42 | 49 |
| `test_source3.tsv` | 2 | 103 | 25.66 ± 9.57 | 25 | 19 | 32 | 38 | 42 | 50 |

### Business Address Length (Characters)

| File | Min | Max | Mean ± Std | Median | P25 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 11 | 256 | 52.07 ± 25.33 | 41 | 33 | 70 | 90 | 103 | 124 |
| `train_source2.tsv` | 8 | 249 | 47.83 ± 23.7 | 37 | 31 | 63 | 84 | 97 | 118 |
| `train_source3.tsv` | 2 | 240 | 48.32 ± 20.18 | 42 | 35 | 55 | 78 | 92 | 116 |
| `test_source1.tsv` | 11 | 268 | 57.21 ± 25.03 | 50 | 36 | 74 | 93 | 105 | 126 |
| `test_source2.tsv` | 5 | 269 | 51.78 ± 24.27 | 43 | 32 | 68 | 87 | 99 | 120 |
| `test_source3.tsv` | 5 | 267 | 50.08 ± 21.44 | 44 | 36 | 59 | 82 | 95 | 118 |

---

## 6. Ground Truth Analysis

- **Total Source 1 Entities in Ground Truth:** 2,206,821
- **Unique Source 1 Entities:** 2,206,821 (Duplicate rows: 0)
- **Singleton Entities (0 matches):** 123,247 (**5.58%**)
- **Total Matched Entity IDs:** 7,638,365
  - **Source 2 Matches:** 3,693,619 (48.36%)
  - **Source 3 Matches:** 3,944,746 (51.64%)
- **Maximum Matches for One Entity:** 11 (Entity `S1-765235386`)

### Match Count Distribution per Source 1 Entity

| Matches | Number of S1 Entities | Percentage | Cumulative % |
|---|---|---|---|
| 0 | 123,247 | 5.58% | 5.58% |
| 1 | 119,157 | 5.40% | 10.98% |
| 2 | 375,212 | 17.00% | 27.98% |
| 3 | 530,841 | 24.05% | 52.03% |
| 4 | 484,115 | 21.94% | 73.97% |
| 5 | 321,957 | 14.59% | 88.56% |
| 6 | 164,868 | 7.47% | 96.03% |
| 7 | 63,968 | 2.90% | 98.93% |
| 8 | 18,680 | 0.85% | 99.78% |
| 9 | 4,205 | 0.19% | 99.97% |
| 10 | 534 | 0.02% | 99.99% |
| 11 | 37 | 0.00% | 99.99% |

---

## 7. Key Findings & Engineering Implications

1. **Zero Raw Missing Values:** Across all 26.4 million records, entity_id, business_name, business_address, and country have 0% null/empty entries.
2. **Zero Primary Key Duplicates:** Every file has 100% unique entity IDs without any duplicate rows.
3. **Clean Tab Separators:** Zero malformed lines across all source tables and ground truth.
4. **Open-Set Country Feature:** Test introduces France (~20% of test records). Models MUST NOT hard-code {US, India}.
5. **Singleton Dominance & High Cardinality:** 5.58% are strict singletons (no match), while 47.96% have 4+ matches (up to 11 matches). A pipeline that forces exactly 1 match per entity will fail severely.
6. **Memory Constraints:** Peak system RAM is ~16 GB with ~1 GB freely available. Full pandas loads will trigger OOM. High-throughput chunked processing, Polars/PyArrow, or memory-mapped arrays must be standard architecture.
