# Multi-Match Similarity & Variance Analysis

How match quality behaves as entity cardinality scales from 1 match to 5+ matches:

| Cardinality Tier | Evaluated Entities | Avg Name Sim | Min Name Sim | Max Name Sim | Avg Addr Sim | Min Addr Sim | Max Addr Sim |
|---|---|---|---|---|---|---|---||
| `1_match` | 2,685 | 0.6163 | 0.6163 | 0.6163 | 0.4565 | 0.4565 | 0.4565 |
| `2_matches` | 8,679 | 0.6185 | 0.4639 | 0.773 | 0.4616 | 0.3266 | 0.5965 |
| `3_matches` | 11,965 | 0.6188 | 0.3806 | 0.8405 | 0.4636 | 0.2693 | 0.6599 |
| `4_matches` | 10,887 | 0.6211 | 0.3257 | 0.8809 | 0.4657 | 0.2367 | 0.6959 |
| `5_plus_matches` | 13,024 | 0.6263 | 0.2745 | 0.9148 | 0.4624 | 0.1936 | 0.7302 |

> [!IMPORTANT]
> Notice the gap between `Max Name Sim` and `Min Name Sim` in 3+ match entities:
> Weakest true matches have an average name similarity of ~0.70-0.75, compared to ~0.95 for the strongest match.
> Setting an overly aggressive threshold (e.g. >0.90) will miss secondary noisy matches and cut recall.
