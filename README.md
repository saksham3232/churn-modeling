# ANN Classification - Customer Churn Modeling

This repository contains a deep learning project for predicting customer churn using Artificial Neural Networks (ANNs). The core of the project is a Streamlit web application that predicts whether a customer is likely to leave a bank based on various features.

## Table of Contents

- [Overview](#overview)
- [Files and Structure](#files-and-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Model and Encoders](#model-and-encoders)
- [Dataset](#dataset)
- [License](#license)

## Overview

The goal of this project is to build, tune, and deploy an ANN to classify bank customers as "churn" or "not churn". It includes the full workflow: data preprocessing, model training, hyperparameter tuning, and deployment in a Streamlit app.

## Files and Structure

- `app.py`: Streamlit app for customer churn prediction.
- `regression_app.py`: Streamlit app for salary regression (see notebook below).
- `experiments.ipynb`: Jupyter notebook for ANN experiments and model development.
- `hyperparameter_tuning_ANN.ipynb`: Notebook for tuning ANN hyperparameters.
- `prediction.ipynb`: Notebook for making batch churn predictions.
- `salary_regression.ipynb`: Salary regression notebook.
- `Churn_Modelling.csv`: Main dataset.
- `model.h5`: Trained ANN model for churn prediction.
- `regression_model.h5`: Trained regression model.
- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`: Preprocessing objects (encoders and scaler).
- `requirements.txt`: Python dependencies.

[Browse all files in the repository.](https://github.com/saksham3232/churn-modeling/blob/main/)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saksham3232/churn-modeling.git
   cd churn-modeling
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open the app in your browser** (Streamlit will provide a local URL).

## Requirements

See `requirements.txt` for the full list. Main libraries include:
- streamlit
- tensorflow
- scikit-learn
- pandas
- numpy

## Usage

The web app collects user data (geography, gender, age, balance, credit score, etc.), preprocesses the input using stored encoders and scalers, and predicts the probability of churn using the trained ANN model. If the probability exceeds 0.5, the customer is considered likely to churn.

## Notebooks

- Use `experiments.ipynb` for model development and evaluation.
- Use `hyperparameter_tuning_ANN.ipynb` to see the hyperparameter optimization workflow.
- Use `prediction.ipynb` and `salary_regression.ipynb` for additional analysis and regression tasks.

## Model and Encoders

- `model.h5`: Main churn classification model.
- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`: Encoders for gender and geography.
- `scaler.pkl`: Feature scaler used during model training.
- `regression_model.h5`: Model for salary regression.

## Dataset

- `Churn_Modelling.csv`: Includes customer features and churn labels.

---

*Note: Only the top 10 files are listed due to API limitations. Visit the [repository on GitHub](https://github.com/saksham3232/churn-modeling/blob/main/) to view all files.*
