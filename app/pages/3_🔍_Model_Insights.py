import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

st.set_page_config(page_title="Model Insights", page_icon="🔍", layout="wide")

st.title("🔍 Model Insights & Performance")
st.markdown("*Understanding how the model works and what drives predictions*")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model Performance Section
st.subheader("📊 Model Performance Comparison")

perf_data = {
    'Model': ['CatBoost Tuned', 'XGBoost Default', 'LightGBM Default', 'Random Forest', 'Gradient Boosting'],
    'R² Score': [0.7519, 0.7385, 0.7244, 0.6826, 0.7190],
    'RMSE ($)': [86.67, 88.97, 91.34, 98.03, 92.23],
    'MAE ($)': [42.88, 43.34, 45.00, 48.37, 45.84]
}
perf_df = pd.DataFrame(perf_data)

col1, col2 = st.columns([3, 2])

with col1:
    st.dataframe(
        perf_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
               .highlight_min(subset=['RMSE ($)', 'MAE ($)'], color='lightyellow'),
        use_container_width=True, 
        hide_index=True
    )

with col2:
    st.markdown("### 🏆 Best Model")
    st.metric("CatBoost R² Score", "0.7519 (75.2%)", delta="+1.3% vs XGBoost")
    st.metric("RMSE (Avg Error)", "$86.67")
    st.metric("MAE (Median Error)", "$42.88")

# Feature Importance
st.markdown("---")
st.subheader("🔑 Top 10 Most Important Features")

feature_imp = pd.DataFrame({
    'Feature': ['Room Type: Private', 'Minimum Nights', 'Accommodates', 
                'Longitude', 'Distance to Center', 'Bedrooms', 'Latitude',
                'Amenities Count', 'Bathrooms', 'Host Listings'],
    'Importance': [11.76, 11.41, 9.02, 7.23, 7.02, 4.71, 2.99, 2.70, 2.54, 2.29]
})

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.viridis_r(np.linspace(0.2, 0.8, 10))
bars = ax.barh(range(10), feature_imp['Importance'].values[::-1], color=colors[::-1], edgecolor='white', linewidth=1.5)
ax.set_yticks(range(10))
ax.set_yticklabels(feature_imp['Feature'].values[::-1], fontsize=11)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Important Features for Price Prediction', fontsize=14, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for bar, val in zip(bars, feature_imp['Importance'].values[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

st.pyplot(fig)

# Key Insights
st.markdown("---")
st.subheader("💡 Key Insights from the Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea15, #764ba215); 
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; height: 100%;">
        <h4>🏠 Property Features Matter Most</h4>
        <ul>
            <li><b>Room type</b> is the #1 predictor — entire homes cost significantly more</li>
            <li><b>Accommodation capacity</b> (beds, bedrooms, bathrooms) directly drives price</li>
            <li><b>Minimum nights</b> has surprising importance — shorter minimums = higher rates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #764ba215, #667eea15); 
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid #764ba2; height: 100%;">
        <h4>🗺️ Location is Critical</h4>
        <ul>
            <li><b>Longitude & distance to Manhattan center</b> are top geographic features</li>
            <li><b>Manhattan properties</b> command 2-3x premium over outer boroughs</li>
            <li><b>Amenities count</b> positively correlates but with diminishing returns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Model Details
st.subheader("🤖 Model Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h3 style="color: #667eea;">CatBoost</h3>
        <p>Gradient Boosting on Decision Trees</p>
        <p><b>1000 iterations</b></p>
        <p><b>Depth: 8</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h3 style="color: #667eea;">Optuna Tuning</h3>
        <p>Bayesian Hyperparameter Optimization</p>
        <p><b>30 trials</b></p>
        <p><b>5-fold CV</b></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h3 style="color: #667eea;">Features</h3>
        <p>Engineered from raw data</p>
        <p><b>110 features</b></p>
        <p><b>21,200 listings</b></p>
    </div>
    """, unsafe_allow_html=True)

# Learning Points
st.markdown("---")
st.subheader("📚 What This Project Demonstrates")

skills = [
    "✅ End-to-end ML pipeline (data collection → deployment)",
    "✅ Advanced feature engineering (NLP-ready, geospatial, statistical)",
    "✅ Hyperparameter tuning with Optuna (Bayesian optimization)",
    "✅ Multiple model comparison (CatBoost, XGBoost, LightGBM, Ensemble)",
    "✅ Interactive web app with Streamlit (3-page dashboard)",
    "✅ Model explainability through feature importance",
    "✅ Production-ready code structure and deployment"
]

for skill in skills:
    st.markdown(f"- {skill}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🔍 Powered by CatBoost | Trained on 21,200+ NYC Airbnb listings | © 2025</p>
</div>
""", unsafe_allow_html=True)