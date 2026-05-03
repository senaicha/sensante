from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ---------------- APP ----------------
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# ---------------- MODELE ----------------
print("Chargement du modele...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")

# ---------------- SCHEMAS ----------------
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool
    fatigue: bool
    maux_tete: bool
    region: str


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# ---------------- HEALTH ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

# ---------------- PREDICT ----------------
@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # 1. Encodage
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prediction
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 4. Confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Recommandations
    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }

    # 6. Retour
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )