# cinema-prediction-ML
This project focuses on predicting cinema attendance based on historical data and movie metadata collected via The Movie Database (TMDb) API. It is part of a broader initiative to apply machine learning techniques to real-world forecasting problems.

## Project Overview

- Programming Language: Python
- Frameworks/Libraries: Pandas, NumPy, Scikit-Learn, TensorFlow/Keras
- Dataset: Five years of cinema data (local CSV), enriched via API
- Objective: Predict the total number of attendees (column `TOT`) for a given movie screening

## Project Structure

- `data/`: Contains the raw CSV file and scripts for API-based enrichment
- `notebooks/`: Jupyter Notebooks used for exploratory analysis and prototyping
- `src/`: Modular Python scripts for data loading, preprocessing, modeling, and inference
- `models/`: Saved trained models (e.g., .pkl, .h5)
- `README.md`: Project documentation
- `requirements.txt`: Required dependencies

## Machine Learning Pipeline

1. **Data Preparation**
   - Convert `total` column to numeric and drop missing values
   - Select relevant numerical features
   - Apply one-hot encoding for categorical variables (e.g., weekday)

2. **Modeling**
   - Train/test split with `train_test_split`
   - Model 1: Random Forest Regressor
   - Model 2: Neural Network (Keras Sequential API)
   - Evaluation Metrics: Mean Squared Error (MSE), R² Score

3. **Execution**
   - Models can be trained and evaluated via modular scripts
   - Output includes performance plots and predictions vs ground truth

## Key Skills Demonstrated

- Data cleaning and transformation
- Use of REST APIs for data enrichment
- Training and evaluation of machine learning models
- Modular Python code design
- Use of Jupyter Notebooks for analysis
- Git/GitHub project organization and documentation

## How to Run

```bash
git clone https://github.com/yourusername/cinema-attendance-predictor.git
cd cinema-attendance-predictor
pip install -r requirements.txt

# Step 1: Prepare data
python src/data_loader.py

# Step 2: Train model
python src/model.py

# Step 3: Make predictions
python src/predict.py
