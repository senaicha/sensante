import pandas as pd

df = pd.read_csv("data/patients_dakar.csv")

print("Nombre de patients :", len(df))
print("Colonnes :", list(df.columns))

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Répartition des diagnostics ---")
print(df["diagnostic"].value_counts())

print("\n--- Température moyenne par diagnostic ---")
print(df.groupby("diagnostic")["temperature"].mean())