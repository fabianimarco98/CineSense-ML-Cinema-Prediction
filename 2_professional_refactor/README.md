# 🎬 Cinema ML Project – Analisi e Previsioni Ingressi Cinema

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/luigideidda/cinema-ml-project)

Questo progetto analizza e prevede gli ingressi al cinema utilizzando tecniche di Machine Learning integrate con dati meteorologici e informazioni sui film.

## 📊 Obiettivi
- Analisi esplorativa degli ingressi (EDA)
- Visualizzazione di tendenze settimanali, stagionali e climatiche
- Previsioni tramite modelli di regressione avanzati
- Integrazione di dati meteo (Open-Meteo API) e dati film (TMDb API)

## 🗂️ Struttura del progetto

📁 input/ → Dati originali (CSV/XLSX)
📁 grafici/ → Grafici generati durante l’EDA
📁 modelli/ → Modelli ML salvati (.pkl)
📄 Grafici_corretti.py → Visualizzazioni e analisi
📄 preprocessing.py → Pulizia dati + meteo + TMDb
📄 train_models.py → Addestramento e selezione modelli
📄 requirements.txt → Librerie Python richieste
📄 README.md → Questo file


## ⚙️ Setup ambiente

### Requisiti
Python ≥ 3.9  
Librerie principali:  
`pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `catboost`, `joblib`, `requests`, `holidays`

### Installazione
```bash
pip install -r requirements.txt

📡 Dati esterni
Meteo: Open-Meteo API

Film: TMDb API (richiede una chiave API)


🤖 Modelli ML utilizzati
Linear Regression

Ridge Regression

Random Forest Regressor

Gradient Boosting Regressor

CatBoost Regressor

XGBoost Regressor

SVR (Support Vector Regression)

✔️ Le feature sono selezionate via RFE
✔️ I modelli migliori vengono salvati in modelli/
✔️ Valutazione con R2, MAE, RMSE

📈 Grafici generati
Ingressi per giorno della settimana

Ingressi per stagione

Correlazione pioggia / temperatura / ingressi

Boxplot per fascia oraria, festività, weekend

Tutti i grafici vengono salvati nella cartella grafici/.

🧠 Autore
Luigi Deidda
📧 [inserisci email o contatto se vuoi]

📄 Licenza
MIT – Sentiti libero di usare, modificare e condividere questo progetto.
