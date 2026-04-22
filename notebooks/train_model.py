import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Nettoyer les colonnes
df.columns = df.columns.str.strip()

# Vérification
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print("\nDiagnostics :")
print(df['diagnostic'].value_counts())

# Encodage
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Features et target
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print("\nFeatures shape :", X.shape)
print("Target shape :", y.shape)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nEntrainement :", X_train.shape[0], "patients")
print("Test :", X_test.shape[0], "patients")

# Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModèle entraîné !")
print("Nombre d'arbres :", model.n_estimators)
print("Nombre de features :", model.n_features_in_)
print("Classes :", list(model.classes_))

# Prédictions
y_pred = model.predict(X_test)

comparison = pd.DataFrame({'Vrai diagnostic': y_test.values[:10], 'Prediction': y_pred[:10]})
print("\nComparaison :")
print(comparison)

# Métriques
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(cm)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Prédiction du modèle")
plt.ylabel("Vrai diagnostic")
plt.title("Matrice de confusion - SenSante")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.show()

import joblib
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille
size = os.path.getsize("models/model.pkl")

print("\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder les features (optionnel mais propre)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")
import joblib

# Charger depuis les fichiers
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")
# Nouveau patient
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

# Encoder les variables
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer les features
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

# Prédiction
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")

print("\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")