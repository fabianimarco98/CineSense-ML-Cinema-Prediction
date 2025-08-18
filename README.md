# Project CineSense: A Machine Learning Application

This repository documents the development journey of CineSense, a machine learning project focused on predicting cinema attendance based on historical data and movie metadata collected via The Movie Database (TMDb) and weather APIs.

## Project Journey & Purpose

This project began as a self-learning exercise to deepen my practical skills in machine learning and data analysis. The development is structured in three main phases:

1.  **Initial Prototype:** The first iteration of the code, which I developed with the assistance of AI tools, can be found in the [`1_initial_prototype/`](./1_initial_prototype/) directory. This version represents my initial approach to data processing and modeling.

2.  **Professional Collaboration & Refactor:** To benchmark my work against professional standards and learn advanced best practices, I collaborated with professional developer **Luigi Deidda**. His more robust and modular version of the code, which served as a crucial learning benchmark, is located in the [`2_professional_refactor/`](./2_professional_refactor/) directory.
    > **Attribution:** The code within the `2_professional_refactor` directory was developed by Luigi Deidda as part of a professional collaboration. His original repository can be found [here](https://github.com/luigideidda/cinema-ml-project). My contribution was the integration and analysis of his code, as well as the development of the initial prototype and all subsequent work in the `3_my_further_work` directory.

3.  **My Further Work & Analysis:** Building on the professional codebase, I continued the project's development. My work, found in the [`3_my_further_work/`](./3_my_further_work/) directory, involved refactoring the scripts into a Jupyter Notebook environment for enhanced readability and interactive analysis. I also implemented a deep learning model using Ludwig (a PyTorch-based library) to explore more complex, non-linear relationships in the data.

## My Contributions & Key Learnings
* **Project Initiation & Scoping:** Defined the project goals and initial data exploration.
* **Initial Prototyping:** Developed the first functional model using Scikit-learn.
* **Code Integration & Analysis:** Integrated the professionally written code into a Jupyter Lab environment for analysis and learning.
* **Model Expansion:** Implemented and tested a deep learning model using Ludwig (PyTorch) to improve predictive performance.
* **Key Skills Applied:** Data Preprocessing, Feature Engineering, Model Evaluation, Python (Pandas, Scikit-learn, TensorFlow/Keras, PyTorch/Ludwig).

---

## Technical Overview

* **Programming Language:** Python
* **Frameworks/Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Ludwig (PyTorch)
* **Dataset:** Five years of cinema data (local CSV), enriched via TMDb and weather APIs.
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
    * **Model 3:** Deep Learning Model (Ludwig/PyTorch)
    * **Evaluation Metrics:** Mean Squared Error (MSE), R² Score

### How to Run

```bash
# Clone the repository
git clone [https://github.com/fabianimarco98/CineSense.git](https://github.com/fabianimarco98/CineSense.git)
cd CineSense

# Install dependencies
pip install -r requirements.txt

# Instructions for running scripts are available within each project folder.
