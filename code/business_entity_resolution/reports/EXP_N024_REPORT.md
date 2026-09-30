# EXP_N024 — Supervised Entity Representation Learning & Recursive Latent Frontier Decoder Report

## 1. Executive Summary & Authoritative Milestone

In **EXP_N024**, we addressed the core bottleneck identified in N023: representation quality and local ranking degradation across deep multi-match entity clusters. We introduced **Supervised Multi-View Entity Representation Learning** combined with a **Multi-Stage Recursive Latent Frontier Decoder**.

### The Breakthrough Result
- **Previous Champion (EXP_N023):** Complete-System Macro $F_{0.5} = \mathbf{0.985939}$
- **New Champion (EXP_N024):** Complete-System Macro $F_{0.5} = \mathbf{0.986756}$
- **Absolute Gain:** $\mathbf{+0.000817}$ (consistent authoritative step toward the 0.990 milestone)
- **False Positives (FP):** Reduced from $198 \to \mathbf{192}$ (**-6 FPs eliminated!**)
- **False Negatives Model (in pool):** Reduced from $2,121 \to \mathbf{1,879}$ (**242 true matches rescued!**)
- **Total False Negatives:** Reduced from $2,525 \to \mathbf{2,283}$
- **Macro Precision:** Increased to $\mathbf{0.992543}$
- **Macro Recall:** Increased to $\mathbf{0.973212}$

---

## 2. Complete Authoritative Trajectory

| Experiment | Architecture / Strategy | Complete-System Macro $F_{0.5}$ (Metric B) | Decision Layer $F_{0.5}$ (Metric A) | Macro Precision | Macro Recall | False Positives (FP) | In-Pool Model FN | Blocker Miss FN | Total FN |
|---|---|---|---|---|---|---|---|---|---|
| **N019-F** | HierGAT+ Collective Alignment | 0.984094 | 0.980655 | 0.993438 | 0.962092 | 103 | 2,925 | 404 | 3,329 |
| **N020** | Selective Graph Decoder | 0.984281 | 0.980845 | 0.993552 | 0.962537 | 103 | 2,869 | 404 | 3,273 |
| **N021** | Corruption-Aware Rescue | 0.984264 | 0.980829 | 0.993524 | 0.962567 | 105 | 2,866 | 404 | 3,270 |
| **N022** | Listwise Local Gap Decoder | 0.984299 | 0.980863 | 0.993552 | 0.962585 | 103 | 2,865 | 404 | 3,269 |
| **N023** | Learned Prefix Boundary Model | 0.985939 | 0.982555 | 0.992453 | 0.970758 | 198 | 2,121 | 404 | 2,525 |
| **EXP_N024** | **Supervised Representation & Recursive Frontier (NEW CHAMPION)** | **0.986756** | **0.983388** | **0.992543** | **0.973212** | **192** | **1,879** | **404** | **2,283** |

---

## 3. Mandated Ablation Matrix Results

| Ablation | Configuration | Metric B ($F_{0.5}$) | Precision | Recall | False Positives | Total FN |
|---|---|---|---|---|---|---|
| **N024-A** | N023 Champion Baseline | 0.985939 | 0.992453 | 0.970758 | 198 | 2,525 |
| **N024-B** | Latent Entity Memory Projection ($\tau = 0.75$) | 0.986490 | 0.992793 | 0.971645 | 167 | 2,439 |
| **N024-C** | S2 $\leftrightarrow$ S3 Relational Support Conditioning | 0.986527 | 0.992590 | 0.972288 | 186 | 2,392 |
| **N024-D** | Hard-Negative Contrastive Penalization Guard | 0.986082 | 0.992797 | 0.970347 | 168 | 2,543 |
| **N024-E** | Recursive Pass 1 Frontier Model ($\tau_1 = 0.70$) | 0.986561 | 0.992584 | 0.972460 | 187 | 2,364 |
| **N024-F** | 2-Stage Recursive Entity Frontier Decoder ($\tau_1=0.70, \tau_2=0.75$) | 0.986731 | 0.992561 | 0.973059 | 190 | 2,301 |
| **N024-G** | **Full 3-Stage Recursive Latent Frontier Decoder ($\tau_1=0.70, \tau_2=0.75, \tau_3=0.70$)** | **0.986756** | **0.992543** | **0.973212** | **192** | **2,283** |

---

## 4. Key Architectural Discoveries

1. **Why Supervised Latent Representation Broken the Single-Pass Frontier:**
   - In N023, boundary decisions evaluated candidates in isolation or only relative to S1 and nearest neighbors.
   - N024 models the entity as a dynamically evolving latent cluster: as soon as high-confidence candidates are accepted, the candidate pool features are enriched with dynamic cosine variance, max/min accepted cosine distance, and accepted sibling cluster centroid agreement.
   - This provided the discriminator with decisive separation power, allowing candidates at rank 3–12 with lower base probability ($p \in [0.40, 0.75]$) to be admitted because they demonstrated high semantic affinity to confirmed entity siblings.

2. **Recursive Multi-Stage Decoding:**
   - Single-stage thresholding was blind to true matches lying behind deep clusters (e.g. entities with 4–8 true matches).
   - By executing a multi-pass recursive frontier expansion, Pass 1 admitted 926 true matches, which formed an expanded anchor set for Pass 2 to safely rescue another 63 true matches, and Pass 3 another 18 true matches, all while strictly capping false positive admissions.

---

## 5. Artifacts and Reproduction
- **Champion Artifact:** [`reports/champion.json`](file:///C:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/reports/champion.json)
- **Standalone Reproducible Script:** [`scripts/run_n024.py`](file:///C:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/scripts/run_n024.py) (Runtime: 10.3 seconds)
- **Experiment Evaluation Summary:** [`experiments/EXP_N024/evaluation_summary.json`](file:///C:/Users/ronit/Videos/hackathons/AmazonML/amazon-ml-challenge/code/business_entity_resolution/experiments/EXP_N024/evaluation_summary.json)
