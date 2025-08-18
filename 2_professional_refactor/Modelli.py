from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
#test

# Crea cartella "modelli" se non esiste
model_dir = os.path.join(os.path.dirname(__file__), "modelli")
os.makedirs(model_dir, exist_ok=True)

# ------------------ 📥 Load & Clean Data ------------------
path = "input/dati_cinema_updated.csv"
df = pd.read_csv(path, encoding="UTF-8", sep=",", dayfirst=True, on_bad_lines="skip")

target = "reduced"  # si può modificare a piacimento con "full_price" o "reduced"

colonne_da_escludere = [
    "full_price", "reduced", "free", "total",       # biglietti
    "title", "date", "time", "datetime", "data_str",  # metadati
    "cast", "director", "keywords", "genres"          # testo non gestito
]

df["weekend"] = df["weekend"].astype(int)

# Riempie eventuali NaN nei numerici
num_cols = ["temp_max", "temp_min", "precip_mm", "vote_average", "popularity"]
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# ------------------ 🧪 Log-transform delle feature sbilanciate ------------------
X = df.drop(columns=colonne_da_escludere + [target])

numeriche = X.select_dtypes(include=np.number).columns.tolist()
skewness = X[numeriche].skew().sort_values(ascending=False)
log_cols = skewness[skewness > 1].index.tolist()

print(f"\n📈 Variabili trasformate con log1p (skew > 1): {log_cols}")
X[log_cols] = X[log_cols].apply(np.log1p)

# Salva log-colonne trasformate per futura inference
joblib.dump(log_cols, os.path.join(model_dir,f"log_transformed_features_{target}.pkl"))

# ------------------ 🎯 Target (solo log su reduced)
if target == "reduced":
    y = np.log1p(df[target])
else:
    y = df[target]

# ------------------ 🔣 Categorical encoding
cat_cols = ["fascia_oraria", "giorno_settimana", "stagione", "festività"]
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ------------------ 🔀 Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("✔️ Preprocessing completato. Pronto per il modello!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# ------------------ 📦 Modelli con RFE ------------------
modelli = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting (Tuned)": GradientBoostingRegressor(random_state=42),
    "CatBoost (Tuned)": CatBoostRegressor(silent=True, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "Ridge Regression": Ridge(),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
}

results = []

for nome, modello in modelli.items():
    print(f"\n🔍 Selezione feature per: {nome}")

    if nome == "SVR":
        rfe = RFE(estimator=LinearSVR(max_iter=10000), n_features_to_select=5)
    else:
        rfe = RFE(estimator=modello, n_features_to_select=5)

    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    print(f"➡️ Features selezionate: {selected_features}")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # ------------------ 🔧 Tuning + Salvataggio ------------------
    filename_base = nome.lower().replace(" ", "_").replace("(", "").replace(")", "")

    if nome == "SVR":
        modello.fit(X_train_sel, y_train)
        pred = modello.predict(X_test_sel)


    elif "CatBoost" in nome:
        param_grid = {
            'iterations': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
            'bagging_temperature': [0.0, 0.2, 0.5, 1.0]
        }
        search = RandomizedSearchCV(modello, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train_sel, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(X_test_sel)

    elif "Gradient Boosting" in nome:
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }
        search = RandomizedSearchCV(modello, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train_sel, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(X_test_sel)


    elif "XGBoost" in nome:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        search = RandomizedSearchCV(modello, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train_sel, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(X_test_sel)


    else:
        modello.fit(X_train_sel, y_train)
        pred = modello.predict(X_test_sel)


    results.append({
        "Modello": nome,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "Features": selected_features,
        "ModelObject": best_model if 'best_model' in locals() else modello,
        "FilenameBase": filename_base + f"_{target}"
    })

# ------------------ 📋 Riepilogo finale e salvataggio top 3 ------------------
results_df = pd.DataFrame(results)
top3 = results_df.sort_values(by="R2", ascending=False).head(3)

print(f"\n📋 Riepilogo finale dei top 3 modelli per previsione '{target}':")
print(top3[["Modello", "MAE", "RMSE", "R2", "Features"]].to_string(index=False))

# Salva i top 3 modelli e le rispettive features
for _, row in top3.iterrows():
    joblib.dump(row["ModelObject"], os.path.join(model_dir, f"{row['FilenameBase']}.pkl"))
    joblib.dump(row["Features"], os.path.join(model_dir, f"{row['FilenameBase']}_features.pkl"))
