import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
code_dir = os.path.join(project_root, "code", "business_entity_resolution")
sys.path.insert(0, os.path.join(code_dir, "src"))
sys.path.insert(0, os.path.join(code_dir, "scripts"))

from run_phase5_experiment import load_train_and_val_samples

dataset_dir = os.path.join(project_root, "dataset")
validation_ids_path = os.path.join(code_dir, "experiments", "validation_s1_ids.txt")

t0 = time.time()
print("Starting load_train_and_val_samples...", flush=True)
train_s1, train_target, train_matches, val_s1, val_target, val_matches = load_train_and_val_samples(
    dataset_dir, validation_ids_path, val_sample_size=2000, train_sample_size=1000, seed=42
)
print(f"LOADED in {time.time()-t0:.2f}s: {len(train_s1)} train S1, {len(val_s1)} val S1, {len(train_target)} targets!", flush=True)
