import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts")))
import fp_guard_forensics as fg

print(f"Total df_fps from length-2 expansion: {len(fg.df_fps)}")
for idx, row in fg.df_fps.iterrows():
    print(f"FP {idx:2d}: S1={row['s1_id']}, cand={row['cand_id']}, r={int(row['r'])}, cos={row['neural_cosine']:.3f}, name_j={row['name_jaccard']:.3f}, addr_j={row['address_jaccard']:.3f}, p={row['p']:.6f}")
