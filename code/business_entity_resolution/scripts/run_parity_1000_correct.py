import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
CODE_DIR = Path(r"c:\Users\ronit\Videos\hackathons\AmazonML\amazon-ml-challenge\code\business_entity_resolution")
DATA_DIR = CODE_DIR.parents[1] / "dataset" / "train"
VAL_IDS = CODE_DIR / "experiments" / "validation_s1_ids.txt"

def main():
    print("Loading 1000 validation IDs...")
    with open(VAL_IDS, "r") as f:
        val_ids = [line.strip() for line in f.readlines()[:1000]]
    val_ids_set = set(val_ids)

    print("Loading train_source1.tsv...")
    df_s1 = pd.read_csv(DATA_DIR / "train_source1.tsv", sep="\t")
    df_s1_1000 = df_s1[df_s1["entity_id"].isin(val_ids_set)]
    df_s1_1000.to_csv(DATA_DIR / "test_source1_1000.tsv", sep="\t", index=False)

    print("Loading train_ground_truth.tsv...")
    df_gt = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t")
    df_gt_1000 = df_gt[df_gt["source1_entity_id"].isin(val_ids_set)]
    
    target_ids = set()
    for _, row in df_gt_1000.iterrows():
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        target_ids.update(matches)
        
    print("Loading train_source2.tsv and train_source3.tsv...")
    df_s2 = pd.read_csv(DATA_DIR / "train_source2.tsv", sep="\t")
    df_s3 = pd.read_csv(DATA_DIR / "train_source3.tsv", sep="\t")
    
    # We must extract all S2/S3 targets that belong to these 1000 entities, PLUS all S2/S3 targets in the same country 
    # Wait, the true pipeline will search over the WHOLE S2/S3 index. If we only extract ground truth targets, 
    # the candidate generation (FAISS/Blocking) will NOT find false positives, and the model probabilities will be skewed!
    # To truly replicate the parity audit on 1000 entities, we should extract ALL S2/S3 records for the countries 
    # that these 1000 entities belong to!
    countries = df_s1_1000["country"].unique()
    print(f"Countries for 1000 entities: {countries}")
    
    df_s2_1000 = df_s2[df_s2["country"].isin(countries)]
    df_s3_1000 = df_s3[df_s3["country"].isin(countries)]
    
    df_s2_1000.to_csv(DATA_DIR / "test_source2_1000.tsv", sep="\t", index=False)
    df_s3_1000.to_csv(DATA_DIR / "test_source3_1000.tsv", sep="\t", index=False)
    
    print("Successfully created test_source1_1000.tsv, test_source2_1000.tsv, test_source3_1000.tsv.")

if __name__ == "__main__":
    main()
