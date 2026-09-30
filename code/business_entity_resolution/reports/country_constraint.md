# Empirical Verification: Country as a Hard Blocking Constraint

## 1. Ground Truth Verification Summary

Every single ground-truth match pair was verified against `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`:

- **Total True Match Pairs Verified:** **7,638,365**
- **Cross-Country Matches Found:** **0 (EXACTLY ZERO)**
- **Cross-Country Match Percentage:** **0.000000%**
- **US S1 entities matched outside US:** 0
- **India S1 entities matched outside India:** 0

## 2. Engineering Implication

> [!IMPORTANT]
> **Country is an empirically proven strict invariant.** Records NEVER match across countries.
> Therefore, candidate generation can strictly partition candidate spaces by country:
> - Candidate pairs $(s_1, s_2)$ or $(s_1, s_3)$ where $s_1.country \ne s_{2/3}.country$ are mathematically impossible in ground truth.
> - Hard blocking by country eliminates **~48% to 60% of all potential cross-comparisons** with 0% recall loss.
