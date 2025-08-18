# Project CineSense: A Machine Learning Application

This repository documents the development journey of CineSense, a machine learning project focused on predicting cinema attendance based on historical data and movie metadata collected via The Movie Database (TMDb) API.

## Project Journey & Purpose

This project began as a self-learning exercise to deepen my practical skills in machine learning and data analysis.

1.  **Initial Version:** The first iteration of the code, which I developed with the assistance of AI tools, can be found in the [`1_my_initial_version/`](./1_my_initial_version/) directory. This version represents my initial approach to solving the problem.

2.  **Professional Collaboration & Integration:** To benchmark my work against professional standards and learn advanced best practices, I collaborated with an external developer. The more robust and refactored code provided by the developer, which I have subsequently studied and integrated into a Jupyter environment, is located in the [`2_professional_integration/`](./2_professional_integration/) directory.

My primary role in this second phase was to define the problem, integrate the new code, and analyze its structure and methods to learn from a professional workflow.

## My Contributions & Key Learnings
* **Project Initiation & Scoping:** Defined the project goals and initial data exploration.
* **Initial Prototyping:** Developed the first functional model.
* **Code Integration & Analysis:** Integrated the professionally written code into a Jupyter Lab environment for analysis and learning.
* **Key Skills Applied:** Data Preprocessing, Feature Engineering, Model Evaluation, Python (Pandas, Scikit-learn, TensorFlow/Keras).

---

## Technical Overview

* **Programming Language:** Python
* **Frameworks/Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow/Keras
* **Dataset:** Five years of cinema data (local CSV), enriched via TMDb API
* **Objective:** Predict the total number of attendees (`total` column) for a given movie screening.

### Machine Learning Pipeline

1.  **Data Preparation**
    * Convert `total` column to numeric and handle missing values.
    * Select relevant numerical and categorical features.
    * Apply one-hot encoding for categorical variables (e.g., weekday).

2.  **Modeling**
    * Train/test split with `train_test_split`.
    * **Model 1:** Random Forest Regressor
    * **Model 2:** Neural Network (Keras Sequential API)
    * **Evaluation Metrics:** Mean Squared Error (MSE), R² Score

### How to Run

```bash
# Clone the repository
git clone [https://github.com/fabianimarco98/CineSense.git](https://github.com/fabianimarco98/CineSense.git)
cd CineSense

# Install dependencies
pip install -r requirements.txt

# (Instructions for running scripts from the '2_professional_integration' folder)
# ...
