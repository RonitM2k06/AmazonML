import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts")))
import pandas as pd
from sim_n034_candidate import unique_s1, s1_acc_map, c_s, bs_s, aj_s, neural_cos_s, p_s, hn_s, gt_full, training_edges

fn35 = pd.read_csv("reports/EXP_N035_FN_AUDIT.csv")
fn35_s1 = set(fn35["s1_id"])
unanchored_fn_s1 = set([s for s in fn35_s1 if len(s1_acc_map.get(s, set())) == 0])

rec_stats = []
for i in range(len(unique_s1)):
    s1 = unique_s1[i]
    if s1 not in unanchored_fn_s1:
        continue
    true_s1 = gt_full.get(s1, set())
    for r in range(0, 10):
        cid = c_s[i, r]
        rec_stats.append({
            "s1": s1,
            "cid": cid,
            "r": r,
            "p": float(p_s[i, r]),
            "cos": float(neural_cos_s[i, r]),
            "aj": float(aj_s[i, r]),
            "hn": float(hn_s[i, r]),
            "bs": float(bs_s[i, r]),
            "is_true": cid in true_s1
        })

df_u = pd.DataFrame(rec_stats)
print(f"Total candidate observations: {len(df_u)}, True matches: {df_u['is_true'].sum()}")
print("\nTrue matches by rank:")
print(df_u[df_u['is_true']]['r'].value_counts())
print("\nTrue matches vs False candidates feature averages:")
print(df_u.groupby('is_true')[['p', 'cos', 'aj', 'hn', 'bs']].mean().round(4))

# Check mutual candidate links among these candidates (observable cross-source edges)
# Pair mutuality: candidate 1 and candidate 2 have observable cross-candidate similarity
print("\nChecking Observable Cross-Candidate Consistency (e.g. S2 and S3 both agree with high cos and aj):")
for min_cos in [0.70, 0.75, 0.80]:
    for min_aj in [0.30, 0.40, 0.50]:
        t_count = sum((df_u["cos"] >= min_cos) & (df_u["aj"] >= min_aj) & (df_u["is_true"]))
        f_count = sum((df_u["cos"] >= min_cos) & (df_u["aj"] >= min_aj) & (~df_u["is_true"]))
        print(f"cos>={min_cos:.2f}, aj>={min_aj:.2f} -> True: {t_count}, False: {f_count}, Net: {t_count - 4*f_count}")
