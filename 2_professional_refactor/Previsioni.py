import joblib
import pandas as pd
import numpy as np
import os

model_dir = os.path.join(os.path.dirname(__file__), "modelli")

# ⚙️ Imposta target e modello
target = "reduced"  # scegliere tra: "full_price" e "reduced"
modello_nome = "catboost_tuned"  # esempio: "catboost_tuned", "random_forest", "xgboost", "gradient_boosting"

# 📦 Carica modello e feature selezionate
model_path = os.path.join(model_dir, f"{modello_nome}_{target}.pkl")
features_path = os.path.join(model_dir, f"{modello_nome}_{target}_features.pkl")
log_feat_path = os.path.join(model_dir, f"log_transformed_features_{target}.pkl")

model = joblib.load(model_path)
top_features = joblib.load(features_path)

# 🧮 Verifica se esistono feature trasformate in log
log_features = []
if os.path.exists(log_feat_path):
    log_features = joblib.load(log_feat_path)
    print(f"📈 Le seguenti feature sono state log-trasformate nel training: {log_features}")

print(f"\n✅ Modello caricato: {modello_nome} per target '{target}'")
print("🧩 Feature attese dal modello:", top_features)

# 📥 Inserisci i dati da tastiera
valori_input = {}
print("\n✍️ Inserisci i valori richiesti per ciascuna feature:")

for feature in top_features:
    valore = input(f" - {feature}: ")
    try:
        valore = float(valore)
        if feature in log_features:
            valore = np.log1p(valore)
    except ValueError:
        pass  # Lascia valori non numerici intatti
    valori_input[feature] = valore

# 🔍 Prepara DataFrame per la previsione
nuovo_dato = pd.DataFrame([valori_input])
nuovo_dato = nuovo_dato.reindex(columns=top_features, fill_value=0)

# 🔮 Previsione
predizione = model.predict(nuovo_dato)

# Se il target era log-trasformato, esegui inverse log
if target == "reduced":
    predizione = np.expm1(predizione)

print(f"\n🎯 Valore previsto di '{target}': {predizione[0]:.2f}")
