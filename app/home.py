import streamlit as st

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Airbnb Price Predictor - NYC",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-card h2 {
        color: #667eea;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #666;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .tech-badge {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 Airbnb Price Predictor</h1>
    <p>Advanced Machine Learning for NYC Rental Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>21,200</h2>
        <p>🏠 Listings Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>75.2%</h2>
        <p>🎯 Model Accuracy (R²)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>$86.67</h2>
        <p>📊 Prediction Error (RMSE)</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>110+</h2>
        <p>🔧 Engineered Features</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Project Overview
st.markdown("## 📖 Project Overview")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("""
    <div class="feature-box">
        <h3>🎯 What This Project Does</h3>
        <p>This end-to-end machine learning project predicts the <b>nightly price of Airbnb listings</b> in New York City 
        based on property characteristics, location, amenities, host information, and review scores.</p>
        <p>Built with <b>CatBoost, XGBoost, and LightGBM</b> models, hyperparameter-tuned with <b>Optuna</b>, 
        and deployed as an interactive <b>Streamlit</b> web application.</p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    <span class="tech-badge">🐍 Python</span>
    <span class="tech-badge">📊 Pandas</span>
    <span class="tech-badge">🤖 CatBoost</span>
    <span class="tech-badge">⚡ XGBoost</span>
    <span class="tech-badge">🌲 LightGBM</span>
    <span class="tech-badge">🎯 Optuna</span>
    <span class="tech-badge">📈 Plotly</span>
    <span class="tech-badge">🗺️ Folium</span>
    <span class="tech-badge">🌐 Streamlit</span>
    <span class="tech-badge">📉 Scikit-learn</span>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How to Use
st.markdown("## 🚀 How to Use This App")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Explore Data
    Navigate to **EDA Dashboard** from the sidebar to explore:
    - Price distributions
    - Geographic maps
    - Feature correlations
    - Amenities analysis
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Predict Prices
    Go to **Predict Price** to:
    - Enter listing details
    - Get instant price predictions
    - See what drives the price
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Understand Model
    Visit **Model Insights** for:
    - Feature importance
    - Model performance
    - Learning curves
    - Error analysis
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Data Source
st.markdown("## 📦 Data Source")
st.markdown("""
<div class="feature-box">
    <p>📍 <b>Inside Airbnb</b> - Public dataset of Airbnb listings in New York City (Updated 2025)</p>
    <p>📊 <b>21,200+ listings</b> after cleaning and preprocessing</p>
    <p>🌍 Source: <a href="http://insideairbnb.com/get-the-data/" target="_blank">insideairbnb.com</a></p>
</div>
""", unsafe_allow_html=True)

# Model Info
st.markdown("## 🏆 Model Performance")

model_data = {
    'Model': ['CatBoost Tuned', 'XGBoost Tuned', 'LightGBM Tuned', 'Stacking Ensemble'],
    'R² Score': [0.7519, 0.7385, 0.7244, 0.7502],
    'RMSE': ['$86.67', '$88.97', '$91.34', '$86.95'],
    'MAE': ['$42.88', '$43.34', '$45.00', '$42.65']
}

import pandas as pd
model_df = pd.DataFrame(model_data)
st.dataframe(model_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🏠 Airbnb Price Predictor | Built with ❤️ using Python & Streamlit</p>
    
</div>
""", unsafe_allow_html=True)