# cinema-prediction-ML
# 🎬 Cinema Attendance Predictor

This project uses machine learning models to **predict cinema attendance** (TOT) based on historical and metadata about movies.

## 🚀 Project Overview

- Language: Python
- Models: Random Forest, Neural Network (Keras/Tensorflow)
- Dataset: Custom dataset from 5 years of cinema logs (df.csv)
- Goal: Predict the number of people that will attend a movie based on features like runtime, genre, release date, and metadata from TMDb API.

## 🧱 Project Structure

- `src/`: Python scripts for loading data, training models, and making predictions
- `notebooks/`: Jupyter Notebook for EDA and prototyping
- `models/`: Trained models
- `data/`: Input data (not the full dataset due to size/privacy)

## 🧠 Machine Learning Pipeline

- **Data Cleaning**: handling missing values, encoding categorical features
- **Feature Engineering**: one-hot encoding of `weekday`, scaling
- **Modeling**: trained and compared Random Forest and Neural Network
- **Evaluation**: R², RMSE, visualizations of prediction vs real

## 🔍 Skills Highlighted

- Python (Pandas, NumPy, Scikit-Learn, TensorFlow/Keras)
- API usage (TMDb API for metadata)
- Model evaluation and tuning
- Code modularization and pipeline separation
- Git & GitHub project organization

## 📊 Example Results

> Include here a plot or table comparing real vs predicted values

## 🛠 How to Run

```bash
pip install -r requirements.txt
python src/data_loader.py
python src/model.py
python src/predict.py
