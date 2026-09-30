import time
t0 = time.time()
print("Importing sklearn...", flush=True)
import sklearn
print(f"sklearn imported in {time.time()-t0:.2f}s", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
print("HistGradientBoostingClassifier imported!", flush=True)
