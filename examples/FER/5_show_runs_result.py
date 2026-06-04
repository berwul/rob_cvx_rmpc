import pathlib
import numpy as np
import pandas as pd

p_curr = pathlib.Path(__file__).parent
p_data = p_curr / "data"

df = pd.read_csv(p_curr / "results" / "runs_stats.csv")

df["name"] = df["name"].map({"ft": "Flexible (ours)", "nom": "Nominal", "nom_star": "Oracle", "rt": "Rigid"})
mask = df["status"] == "success"
df["status"] = mask
df[~mask] = pd.NA
print("Successful runs:")
print(df.groupby("name")["status"].mean())

dt = 1e-2
df_g = df.groupby("name")["ts_goal"].agg(["mean", "std"])
names = ["Flexible (ours)",  "Rigid", "Nominal", "Oracle"]
print(" ")
print("Time to goal stats:")
print(np.round(df_g * dt, 2))
