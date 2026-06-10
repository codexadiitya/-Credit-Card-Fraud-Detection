# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using EDA, class imbalance handling (SMOTE), and multiple ML models including XGBoost.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              # Dataset (download separately)
│
├── credit_card_fraud_detection.ipynb  # Main notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── LICENSE                            # MIT License
```

---

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions | 492 fraud cases (~0.17%)
- **Features:** 30 features (V1–V28 are PCA-transformed, plus `Time` and `Amount`)
- **Target:** `Class` — 0 = Genuine, 1 = Fraud

> Download the dataset from Kaggle and place it inside a `data/` folder as `creditcard.csv`.

---

## 🔍 What This Project Covers

### 1. Exploratory Data Analysis (EDA)
- Class distribution (highly imbalanced)
- Transaction amount and time distributions
- Correlation heatmap
- Density plots and time series analysis
- Statistical t-test between fraud and normal transactions

### 2. Data Preprocessing
- Missing value check
- Feature scaling using `StandardScaler`
- Class imbalance handling using **SMOTE** and **Random Over Sampling**

### 3. Model Training
- **Isolation Forest** — anomaly detection baseline
- **Logistic Regression** — with GridSearchCV hyperparameter tuning
- **XGBoost Classifier** — trained on SMOTE-resampled data

### 4. Model Evaluation
- Classification Report (Precision, Recall, F1-Score)
- Evaluation metric visualization
- SHAP values for model interpretability

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aditydiwan/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:
   ```
   data/creditcard.csv
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE, Random Over Sampling |
| `shap` | Model interpretability |
| `scipy` | Statistical tests |

---

## 📈 Key Results

- XGBoost with SMOTE achieved high recall on fraud class
- SHAP analysis revealed the most impactful features for fraud prediction
- Addressed class imbalance effectively using oversampling techniques

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Aditya Diwan**
- 📧 Email: aditydiwan2005@gmail.com
- 🐙 GitHub: [@aditydiwan](https://github.com/aditydiwan)

---

## ⭐ Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspiration: Real-world financial fraud detection systems
