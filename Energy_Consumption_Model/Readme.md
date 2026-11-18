# Industrial Energy Forecasting & Automated Anomaly Detection System

### An End-to-End Machine Learning Pipeline for Energy Demand Prediction and Abnormal Consumption Detection

**Domain:** Industrial IoT | Energy Analytics | Time-Series Forecasting | Anomaly Detection

---

## 1. Project Overview

Energy consumption forecasting is an important part of modern industrial operations and smart grid management.

Electricity demand continuously changes depending on time of day, day of week, seasonal patterns, and operational activities. Traditional static threshold-based monitoring systems often generate false alarms because they cannot adapt to changing consumption patterns.

This project develops a Machine Learning pipeline that:

- Forecasts future electricity consumption using historical consumption patterns.
- Identifies abnormal energy consumption through prediction residuals.
- Compares multiple forecasting algorithms.
- Optimizes model performance through hyperparameter tuning.
- Evaluates model reliability through ablation studies and seasonal robustness testing.

The primary objective is to build a predictive monitoring system that helps identify unusual electricity consumption patterns and supports better energy management decisions.

## 2. Business Problem

Industrial facilities and electricity distribution systems need reliable energy demand predictions to:

- Improve energy resource allocation.
- Reduce operational costs.
- Identify unexpected consumption patterns.
- Detect potential equipment malfunction.
- Minimize unnecessary operational alerts.
- Improve energy monitoring and planning.

**Problem with traditional monitoring:**

Static threshold-based systems use fixed consumption limits to detect abnormal behavior.

However, electricity demand naturally changes throughout the day and across seasons. A fixed threshold may incorrectly classify normal demand variations as anomalies.

**Proposed Solution:**

Use machine learning models to establish an expected consumption baseline. Compare actual consumption against predicted consumption and identify significant deviations using statistical residual analysis.

## 3. Project Objectives

1. Perform dataset ingestion and quality validation.
2. Analyze electricity consumption patterns through Exploratory Data Analysis.
3. Engineer temporal, lag, and rolling statistical features.
4. Develop and compare multiple forecasting models.
5. Optimize forecasting performance using time-series cross-validation.
6. Implement statistical anomaly detection using a 3-Sigma residual threshold.
7. Evaluate feature importance through ablation experiments.
8. Assess model performance across different seasonal periods.

## 4. Dataset

**Dataset:** ElectricityLoadDiagrams20112014

**Source:** UCI Machine Learning Repository

The dataset contains electricity consumption measurements collected from multiple electricity consumers at 15-minute intervals over several years.

Dataset characteristics:

- Multiple electricity consumers.
- Historical electricity consumption records.
- 15-minute observation frequency in the original dataset.
- Timestamp-based consumption measurements.
- Suitable for time-series forecasting and abnormal consumption analysis.

The notebook processes the dataset into an hourly aggregated consumption series for forecasting.

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Statistical Analysis | ACF, PACF, Residual Analysis |
| Model Optimization | GridSearchCV |
| Validation | TimeSeriesSplit |
| Model Serialization | Pickle |
| Environment | Jupyter Notebook |

## 6. Machine Learning Workflow

The project follows a structured time-series machine learning pipeline.

### Step 1: Data Ingestion & Quality Validation

- Load the electricity consumption dataset.
- Convert timestamps into datetime format.
- Check missing values.
- Identify duplicate timestamps.
- Validate 15-minute observation frequency.
- Analyze active and inactive consumers.
- Examine consumption distributions.

### Step 2: Exploratory Data Analysis

Analyze electricity consumption through:

- Overall consumption trends.
- Hour-of-day consumption patterns.
- Day-of-week variations.
- Daily and weekly aggregation.
- Monthly and seasonal patterns.
- Autocorrelation (ACF).
- Partial Autocorrelation (PACF).

### Step 3: Feature Engineering

**Temporal Features**
- Hour of day.
- Day of week.
- Month.
- Weekend indicator.

**Cyclic Features**

Apply sine and cosine transformations to represent periodic patterns:

- Hour sine and cosine.
- Month sine and cosine.

**Historical Lag Features**
- 1-hour lag.
- 2-hour lag.
- 24-hour lag.
- 168-hour lag.

**Rolling Statistical Features**
- 24-hour rolling mean.
- 168-hour rolling mean.

Rolling features are shifted to prevent target leakage.

### Step 4: Time-Series Train-Test Split

The dataset is divided chronologically:

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 27,917 | 80% |
| Testing | 6,980 | 20% |

Test Period: 16 March 2014 – 1 January 2015.

Random shuffling is avoided to preserve temporal ordering and prevent future information from entering the training process.

### Step 5: Forecasting Models

Five forecasting approaches are evaluated:

1. Seasonal Naive Forecasting
2. Multiple Linear Regression
3. Ridge Regression
4. Random Forest Regressor
5. XGBoost Regressor

Evaluation Metrics:

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- MAPE – Mean Absolute Percentage Error

### Step 6: Hyperparameter Optimization

Two models are optimized:

**Ridge Regression**
- Alpha: 0.1, 1, 10, 100, 1000.

**XGBoost**
- Maximum Depth: 3, 5, 7.
- Learning Rate: 0.05, 0.1.
- Number of Estimators: 100, 150.

Optimization uses GridSearchCV with 3-fold chronological TimeSeriesSplit validation.

### Step 7: Anomaly Detection

Anomaly detection is implemented using forecasting residuals.

Residual:

`Residual = Actual Consumption - Predicted Consumption`

A statistical threshold is calculated using three standard deviations:

`Upper Threshold = Mean Residual + 3 × Standard Deviation`

`Lower Threshold = Mean Residual - 3 × Standard Deviation`

Observations outside these boundaries are flagged as potential anomalies.

This approach identifies abnormal consumption relative to the model's expected baseline rather than relying on fixed consumption limits.

### Step 8: Explainability & Ablation Studies

The project evaluates the contribution of different feature groups.

Three configurations are compared:

- Full optimized XGBoost model.
- Model without temporal features.
- Model without historical lag and rolling features.

### Step 9: Seasonal Robustness Testing

Forecasting performance is evaluated across two seasonal groups:

- Spring & Summer: April–August.
- Autumn & Winter: September–January.

This helps examine whether forecasting accuracy remains relatively stable across different seasonal periods.

## 7. Experimental Results

### Baseline Model Comparison

| Model | MAE (kW) | RMSE (kW) | MAPE |
|---|---:|---:|---:|
| Seasonal Naive | 10,446.56 | 15,226.10 | 18.51% |
| Linear Regression | 6,691.63 | 8,867.66 | 14.25% |
| Ridge Regression | 6,691.63 | 8,867.67 | 14.25% |
| Random Forest | 6,073.92 | 8,048.40 | 12.99% |
| XGBoost | 5,999.74 | 8,051.26 | 12.70% |

*MAPE values use the corrected evaluation implementation, excluding actual values less than or equal to 1 kW.*

### Optimized Model Comparison

| Model | MAE (kW) | RMSE (kW) | MAPE |
|---|---:|---:|---:|
| Baseline Ridge | 6,691.63 | 8,867.67 | 14.25% |
| Optimized Ridge | 6,697.61 | 8,875.87 | 14.26% |
| Baseline XGBoost | 5,999.74 | 8,051.26 | 12.70% |
| Optimized XGBoost | 5,917.51 | 7,876.68 | 12.76% |

**Best model by MAE and RMSE:** Optimized XGBoost.

**Best model by MAPE:** Baseline XGBoost.

## 8. Anomaly Detection Results

The optimized XGBoost model is used to calculate prediction residuals and identify anomalous consumption patterns.

| Metric | Result |
|---|---:|
| Total Test Timestamps | 6,980 |
| Detected Anomalies | 69 |
| Anomaly Percentage | 0.99% |
| Residual Mean | -28.62 |
| Residual Standard Deviation | 7,877.19 |

The 3-Sigma residual method identifies approximately 0.99% of test observations as potential anomalies.

These are statistical flags and require operational investigation before being classified as actual equipment faults or energy theft.

## 9. Ablation Study Results

The ablation experiment evaluates how different feature groups influence forecasting performance.

| Configuration | MAE (kW) | Degradation |
|---|---:|---:|
| Full Optimized XGBoost | 5,917.51 | 0% |
| Without Temporal Features | 6,248.61 | 5.60% |
| Without Historical Features | 15,969.66 | 169.87% |

**Key Observation:**

Historical lag and rolling features have a substantial impact on forecasting performance.

Removing historical features increases MAE significantly, indicating that past consumption behavior is an important predictor of future energy demand.

## 10. Seasonal Robustness Results

| Period | Samples | MAE (kW) | RMSE (kW) | MAPE |
|---|---:|---:|---:|---:|
| Spring & Summer | 3,672 | 6,272.61 | 8,378.32 | 12.62% |
| Autumn & Winter | 2,929 | 5,626.36 | 7,424.14 | 12.88% |

The model demonstrates relatively similar MAPE across the two evaluated seasonal groups.

## 11. Key Findings

1. Historical electricity consumption is an important predictor of future demand.
2. Tree-based models outperform the linear models in MAE and RMSE.
3. XGBoost provides strong forecasting performance.
4. Hyperparameter optimization improves XGBoost MAE and RMSE.
5. Removing historical features significantly degrades forecasting accuracy.
6. Residual-based anomaly detection provides a dynamic alternative to static thresholds.
7. Seasonal evaluation helps assess model stability under changing consumption patterns.

## 12. Project Structure

```text
Industrial-Energy-Forecasting/
│
├── Energery_Forcasting_Model.ipynb
│
├── README.md
│
├── requirements.txt
│
├── models/
│   └── xgboost_production_model.pkl
│
└── results/
    ├── forecasting_results/
    ├── anomaly_detection/
    └── visualizations/
```

*The structure above represents the recommended repository organization. The notebook currently exports the serialized XGBoost model as `xgboost_production_model.pkl`.*

## 13. Installation & Execution

### Clone Repository

```bash
git clone https://github.com/Krish-void/Machine-Learning-Project.git

cd  Machine-Learning-Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

Open `Energery_Forcasting_Model.ipynb` and execute the notebook cells in sequence.

The notebook downloads the dataset through KaggleHub.

## 14. Future Improvements

Potential future development directions:

- Build a FastAPI-based forecasting service.
- Develop real-time data ingestion pipelines.
- Integrate PostgreSQL for historical energy records.
- Add automated anomaly alerting.
- Implement model monitoring and drift detection.
- Introduce explainability using SHAP.
- Develop a monitoring dashboard.
- Deploy the forecasting pipeline using Docker and cloud infrastructure.
- Evaluate forecasting performance using rolling-origin backtesting.

## 15. Disclaimer

This project is an experimental machine learning and research implementation based on historical electricity consumption data.

Detected anomalies represent statistically unusual deviations from predicted consumption. They do not independently establish equipment failures, unauthorized usage, or other specific operational causes.

Further validation using real operational data and domain expertise is required before production deployment.

---

**Project Focus:** Time-Series Forecasting | Predictive Analytics | Statistical Anomaly Detection | Industrial Energy Intelligence
