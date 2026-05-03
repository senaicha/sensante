import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib
import os

# =========================
# 1. CHARGEMENT DATASET
# =========================
df = pd.read_csv("data/patients_dakar.csv")
df.columns = df.columns.str.strip()

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print("\nDiagnostics :")
print(df['diagnostic'].value_counts())

# =========================
# 2. ENCODAGE
# =========================
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# =========================
# 3. FEATURES
# =========================
feature_cols = [
    'age', 'sexe_encoded', 'temperature',
    'tension_sys', 'toux', 'fatigue',
    'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

# =========================
# 4. TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape[0])
print("Test:", X_test.shape[0])

# =========================
# 5. MODELE
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModèle entraîné !")

# =========================
# 6. EVALUATION
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy :", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# 7. EXERCICE 1 - IMPORTANCE
# =========================
print("\n--- Importance des features ---")

importances = model.feature_importances_

for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1],
                        reverse=True):
    print(f"{name:20s} : {imp:.3f}")

# =========================
# 8. EXERCICE 2 - PATIENTS
# =========================
print("\n--- Test avec plusieurs patients ---")

patients = [
    {'age': 20, 'sexe': 'F', 'temperature': 36.5, 'tension_sys': 120, 'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},
    {'age': 35, 'sexe': 'M', 'temperature': 40.0, 'tension_sys': 110, 'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar'},
    {'age': 70, 'sexe': 'F', 'temperature': 38.0, 'tension_sys': 130, 'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Dakar'}
]

for p in patients:
    sexe_enc = le_sexe.transform([p['sexe']])[0]
    region_enc = le_region.transform([p['region']])[0]

    features = [
        p['age'], sexe_enc, p['temperature'],
        p['tension_sys'], int(p['toux']),
        int(p['fatigue']), int(p['maux_tete']),
        region_enc
    ]

    pred = model.predict([features])[0]

    print("\nPatient :", p)
    print("Diagnostic :", pred)

# =========================
# 9. VISUALISATION
# =========================
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matrice de confusion")
plt.xlabel("Prediction")
plt.ylabel("Vrai")
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/confusion_matrix.png", dpi=150)

# IMPORTANT : pas de blocage
plt.close()

# =========================
# 10. SAUVEGARDE MODELE
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("\nModèle sauvegardé ✔")

# =========================
# 11. RECHARGEMENT
# =========================
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print("\nModèle rechargé ✔")

# =========================
# 12. TEST FINAL
# =========================
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

pred = model_loaded.predict([features])[0]
proba = model_loaded.predict_proba([features])[0]

print("\n--- Pré-diagnostic ---")
print("Diagnostic :", pred)
print("Probabilité max :", max(proba))