# Air Quality Prediction using Machine Learning

This is a beginner-friendly machine learning project that predicts the Air Quality Index (AQI).

The idea is simple: give the model information about weather, pollution, city/activity data and, when available, a short pollution-event description. The model then estimates the AQI.

## What I used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Gradient Boosting Regression
- Joblib

## Project flow

```text
Dataset
   ↓
Clean the data
   ↓
Prepare features
   ↓
TF-IDF for event text
   ↓
Train Gradient Boosting model
   ↓
Evaluate the model
   ↓
Predict AQI
```

## How to run

Install the packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python train_model.py
```

The program will print:

- MAE
- RMSE
- R² score

It will also create:

```text
air_quality_model.joblib
prediction_results.csv
```

## Important note

The dataset included with this project is for educational/project purposes. If you use official AQI data later, replace the CSV with the real dataset and retrain the model.

## What I learned from this project

This project helped me practice the complete ML workflow:

**data cleaning → feature engineering → NLP → model training → evaluation → prediction**

I also learned how different types of data, including text and numerical values, can be combined in one machine-learning pipeline.
