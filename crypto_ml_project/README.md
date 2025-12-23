# Crypto ML Project

An end-to-end **machine learning project on cryptocurrency price data**, focused on **Bitcoin (BTC-USD)**.  
This project covers the **complete ML pipeline**: data collection, exploratory data analysis (EDA), feature engineering, and model training — built step by step for learning and experimentation.

---

##  Project Objectives

- Collect historical cryptocurrency price data
- Perform exploratory data analysis (EDA)
- Understand return distributions and volatility
- Engineer features for machine learning
- Build a foundation for predictive modeling in crypto markets

---

## 📁 Project Structure
```
crypto_ml_project/
│
├── data/ Raw and processed datasets
│ └── .gitkeep
│
├── notebooks/ Jupyter notebooks for EDA & experiments
│ ├── 01_data_loading_and_eda.ipynb
│ └── .gitkeep
│
├── src/ Source code (data, features, models)
│ └── .gitkeep
│
├── results/ Outputs, plots, and model results
│ └── .gitkeep
│
├── .gitignore
├── README.md

```


---

## Data Source

- **Yahoo Finance**
- Ticker used: `BTC-USD`
- Frequency: **Daily data**
- Downloaded using the `yfinance` Python library

---

## Current Progress

- [x] Project structure setup
- [x] Data download using `yfinance`
- [x] Exploratory Data Analysis (EDA)
- [x] Return distribution visualization
- [ ] Feature engineering
- [ ] ML model training
- [ ] Model evaluation
- [ ] Trading signal generation

---

## Key Concepts Covered

- Financial returns (`pct_change`)
- Data cleaning and NaN handling
- Distribution analysis
- Time-series basics
- ML-ready dataset preparation

---

## Tech Stack

- Python 3.9
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- yfinance
- ta (Technical Analysis library)

---

## Future Work

- Add technical indicators (RSI, MACD, Moving Averages)
- Create classification & regression targets
- Train ML models (Logistic Regression, Random Forest, XGBoost)
- Backtest simple trading strategies
- Evaluate performance using financial metrics

---

## 👤 Author

**Fardwish:)**  
Learning-focused ML & quantitative finance project
