# Name / Address Evidence Matrix

Distribution of true ground-truth matches across Name and Address evidence tiers:

| | Address Strong (Sim ≥ 0.70) | Address Weak (0 < Sim < 0.70) | Address Missing (Empty) |
|---|---|---|---|
| **Name Strong** | **8.64%** (14,943)<br>med_name: 0.913, med_addr: 0.7949 | **27.92%** (48,314)<br>med_name: 0.9118, med_addr: 0.4054 | **2.15%** (3,719)<br>med_name: 0.9091, med_addr: 0.0 |
| **Name Medium** | **7.64%** (13,221)<br>med_name: 0.6522, med_addr: 0.8108 | **21.14%** (36,587)<br>med_name: 0.6667, med_addr: 0.4028 | **1.33%** (2,297)<br>med_name: 0.6818, med_addr: 0.0 |
| **Name Weak** | **5.56%** (9,629)<br>med_name: 0.2895, med_addr: 0.8286 | **24.72%** (42,774)<br>med_name: 0.2, med_addr: 0.375 | **0.9%** (1,549)<br>med_name: 0.2727, med_addr: 0.0 |

> [!NOTE]
> Over **80% of all true matches** sit in the `Name Strong + Address Strong` cell.
> However, ~15-20% of true matches reside in weak/missing address regimes where matching must rely on strong name and token similarity.
