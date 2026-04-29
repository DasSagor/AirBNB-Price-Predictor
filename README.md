# 🏠 Airbnb Price Predictor - NYC

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=for-the-badge&logo=streamlit)
![CatBoost](https://img.shields.io/badge/CatBoost-Tuned-yellow?style=for-the-badge)
![R²](https://img.shields.io/badge/R²-0.752-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**🔗 Live Demo:** [Click Here](https://your-app.streamlit.app)

**End-to-end Machine Learning project for predicting Airbnb nightly prices in New York City.**

---

## 📊 Project Overview

This project builds a complete ML pipeline to predict Airbnb listing prices in NYC using:

- 🏠 **Property characteristics** (room type, bedrooms, bathrooms, amenities)
- 📍 **Location** (borough, coordinates, distance to Manhattan center)
- ⭐ **Host info** (superhost status, response rate, verification)
- 📝 **Reviews** (rating, accuracy, cleanliness, location, value)
- 📅 **Booking details** (minimum nights, availability)

### 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| 📦 Dataset | 21,200+ NYC Airbnb Listings (2025) |
| 🔧 Engineered Features | 110 |
| 🏆 Best Model | CatBoost Tuned |
| 📈 R² Score | **0.752 (75.2%)** |
| 💰 RMSE | **$86.67** |
| 💵 MAE | **$42.88** |

---

## 🏆 Model Performance

| Rank | Model | R² Score | RMSE ($) | MAE ($) |
|:----:|-------|:--------:|:--------:|:-------:|
| 🥇 | **CatBoost Tuned** | **0.7519** | **86.67** | **42.88** |
| 🥈 | Stacking Ensemble | 0.7502 | 86.95 | 42.65 |
| 🥉 | HistGradientBoosting | 0.7445 | 87.94 | 44.67 |
| 4 | Voting Ensemble | 0.7439 | 88.05 | 42.79 |
| 5 | XGBoost | 0.7385 | 88.97 | 43.34 |
| 6 | LightGBM | 0.7244 | 91.34 | 45.00 |
| 7 | Random Forest | 0.6826 | 98.03 | 48.37 |

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn, Folium |
| **ML Models** | CatBoost, XGBoost, LightGBM, Scikit-learn |
| **Hyperparameter Tuning** | Optuna (Bayesian Optimization) |
| **Web App** | Streamlit (Multi-page) |
| **Deployment** | Streamlit Community Cloud (Free) |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

airbnb-price-predictor/
│
├── 📱 app/ # Streamlit Web App
│ ├── home.py # Landing page with overview
│ └── pages/
│ ├── 1_📊_EDA_Dashboard.py # Interactive data visualizations
│ ├── 2_💰_Predict_Price.py # Price prediction tool
│ └── 3_🔍_Model_Insights.py # Model explainability
│
├── 📓 notebooks/ # Jupyter Notebooks
│ ├── 01_initial_eda.ipynb # Phase 1 & 2: Data cleaning & EDA
│ ├── 02_feature_engineering.ipynb # Phase 3: Feature engineering
│ ├── 03_model_training.ipynb # Phase 4: Model training
│ └── 04_hyperparameter_tuning.ipynb # Phase 4.5: Optuna tuning
│
├── 🧠 models/ # Trained Models
│ ├── best_model.pkl # Best model (CatBoost)
│ ├── feature_names.pkl # Feature list for prediction
│ ├── xgboost_tuned.pkl # Tuned XGBoost
│ └── lightgbm_tuned.pkl # Tuned LightGBM
│
├── 📊 data/ # Data files
│ ├── raw/ # Raw CSV from Inside Airbnb
│ └── processed/ # Feature-engineered data
│
├── ⚙️ src/ # Helper scripts
│ └── data_loader.py # Auto-download dataset
│
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # You are here! 📍


---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DasSagor/airbnb-price-predictor.git
cd airbnb-price-predictor

2️⃣ Setup Virtual Environment

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download Data
# Option A: Use the script
python src/data_loader.py

# Option B: Manual download
# Visit: http://insideairbnb.com/get-the-data/
# Download NYC listings.csv.gz → place in data/raw/

5️⃣ Run the App
streamlit run app/home.py
🌐 Open http://localhost:8501 in your browser!


📧 Contact & Links

Platform	Link
💻 GitHub	@DasSagor
🔗 Live Demo	Streamlit App
📊 Dataset	Inside Airbnb
⭐ Show Your Support
If you find this project useful or interesting, please consider:

⭐ Starring this repository

🔀 Forking it for your own use

📢 Sharing it with others

📜 License
This project is open-source and available for learning and portfolio purposes. Data sourced from Inside Airbnb under their terms of use.

<br> <div align="center"> <img src="https://img.shields.io/badge/Built%20with-%E2%9D%A4%EF%B8%8F-red" alt="Built with love"> <br><br> <b>🏠 Airbnb Price Predictor</b><br> <i>End-to-End Machine Learning Project</i><br> © 2025 </div> ``
