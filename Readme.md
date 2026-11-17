# Industrial Energy Forecasting Using Machine Learning

### Time-Series Analysis | Energy Demand Forecasting | Model Optimization

**Project Status:** Ongoing  
**Current Progress:** Week 1–4 Completed | Anomaly Detection In Progress

---

## 1. Project Overview

This project focuses on analyzing historical electricity consumption patterns and developing Machine Learning models for energy demand forecasting.

The primary objective is to understand electricity consumption behavior, identify temporal patterns, build forecasting models, and evaluate their predictive performance.

The project uses historical energy consumption data to establish a data-driven approach for predicting expected electricity demand.

## 2. Dataset Description

**Dataset:** ElectricityLoadDiagrams20112014

| Parameter | Description |
|---|---|
| Total Observations | 140,256 |
| Total Consumers | 370 |
| Sampling Frequency | 15 Minutes |
| Time Period | 2011–2015 |
| Data Type | Time-Series |

### Initial Dataset Analysis

- Identified 370 electricity consumers.
- Identified 202 completely inactive consumers.
- Identified 168 active consumers.
- Performed initial data quality and consumption analysis.

## 3. Work Completed So Far

### Week 1 – Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand the dataset and electricity consumption behavior.

- Dataset structure and data validation.
- Missing values and duplicate timestamp checks.
- Active and inactive consumer identification.
- Consumer-wise descriptive statistics.
- Hourly and weekly consumption patterns.
- Daily, weekly, and monthly consumption analysis.
- Seasonal behavior analysis.
- ACF and PACF analysis to understand historical dependencies.

**Observation:** Electricity consumption exhibits recurring daily, weekly, and seasonal patterns. Different consumers also demonstrate different consumption behaviors.

### Week 2 – Data Preprocessing & Feature Engineering

Prepared the time-series dataset for Machine Learning.

- Temporal feature extraction.
- Cyclic feature transformation using sine and cosine.
- Historical lag feature creation.
- Rolling statistical feature generation.
- Chronological 80:20 train-test splitting.

**Key Features:**
- Hour and Day of Week.
- Month and Weekend Indicator.
- Lag 1h, 2h, 24h and 168h.
- Rolling 24h and 168h averages.

### Week 3 – Baseline Forecasting Models

Implemented and evaluated five forecasting approaches.

1. Seasonal Naive
2. Linear Regression
3. Ridge Regression
4. Random Forest Regressor
5. XGBoost Regressor

Models were compared using:

- MAE – Mean Absolute Error.
- RMSE – Root Mean Squared Error.
- MAPE – Mean Absolute Percentage Error.

### Week 4 – Model Optimization & Comparison

Performed hyperparameter optimization for Ridge Regression and XGBoost using GridSearchCV and TimeSeriesSplit.

Compared baseline and optimized models to evaluate improvements in forecasting performance.

**Model Comparison:**

| Model | MAE (kW) | RMSE (kW) | MAPE (%) |
|---|---:|---:|---:|
| Baseline Ridge | 6691.63 | 8867.67 | 14.25 |
| Optimized Ridge | 6697.61 | 8875.87 | 14.26 |
| Baseline XGBoost | 5999.74 | 8051.26 | 12.70 |
| Optimized XGBoost | 5917.51 | 7876.68 | 12.76 |

**Current Observation:**

Optimized XGBoost achieved better MAE and RMSE compared to its baseline version, while MAPE increased slightly.

This comparison provides the basis for selecting and evaluating the forecasting model in the next project phase.

---

## 4. Current Work – Anomaly Detection

<ins>Week 5 – Currently In Progress</ins>

The current development phase focuses on anomaly detection using forecasting deviations.

The planned approach involves:

- Using the forecasting model to estimate expected electricity consumption.
- Calculating residuals between actual and predicted consumption.
- Analyzing deviations in prediction errors.
- Implementing statistical thresholds to identify potential anomalies.
- Visualizing unusual consumption patterns.

The objective is to identify consumption behavior that significantly deviates from the expected forecasting baseline.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Boosting | XGBoost |
| Statistical Analysis | Statsmodels |
| Optimization | GridSearchCV, TimeSeriesSplit |
| Environment | Jupyter Notebook |

## 6. Repository Structure

```text
Energy-Forecasting/
│
├── Energery_Forcasting_Model.ipynb
├── README.md
└── requirements.txt
```

## 7. Current Project Status

| Project Phase | Status |
|---|---|
| Week 1 – Exploratory Data Analysis | Completed |
| Week 2 – Preprocessing & Feature Engineering | Completed |
| Week 3 – Baseline Model Development | Completed |
| Week 4 – Model Optimization & Comparison | Completed |
| Week 5 – Anomaly Detection | In Progress |

## 8. Project Objective

The broader objective is to develop a Machine Learning-based energy analytics pipeline that combines electricity demand forecasting with the identification of unusual consumption patterns.

**Current Milestone:** Baseline forecasting models have been developed, evaluated, and compared with their optimized versions. Anomaly detection is the current ongoing development phase.

---

**Project Type:** Academic Machine Learning Project  
**Domain:** Energy Analytics | Time-Series Forecasting | Anomaly Detection