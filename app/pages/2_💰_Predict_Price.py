import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Predict Price", page_icon="💰", layout="wide")

st.title("💰 Predict Airbnb Price")
st.markdown("*Enter listing details to get an instant price prediction*")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load model and features
@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "best_model.pkl"
    features_path = PROJECT_ROOT / "models" / "feature_names.pkl"
    
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except FileNotFoundError:
        st.error("❌ Model files not found! Run 04_hyperparameter_tuning.ipynb first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

try:
    model, feature_names = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.stop()

# Create input form
st.markdown("### 🏠 Property Details")
col1, col2 = st.columns(2)

with col1:
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    accommodates = st.number_input("Accommodates (guests)", min_value=1, max_value=16, value=2)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
    beds = st.number_input("Beds", min_value=1, max_value=20, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=10.0, value=1.0, step=0.5)

with col2:
    borough = st.selectbox("Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    neighbourhood = st.selectbox("Neighbourhood (sample)", 
                                ["Hell's Kitchen", "East Village", "Upper West Side", 
                                 "Chelsea", "Midtown", "Financial District"])
    lat = st.number_input("Latitude", value=40.7580, format="%.4f")
    lon = st.number_input("Longitude", value=-73.9855, format="%.4f")

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 📋 Booking Details")
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=2)
    availability_30 = st.number_input("Availability (next 30 days)", min_value=0, max_value=30, value=15)
    instant_bookable = st.selectbox("Instant Bookable", ["Yes", "No"])

with col4:
    st.markdown("### ⭐ Reviews & Host")
    review_score = st.slider("Review Score", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    num_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    is_superhost = st.selectbox("Superhost?", ["Yes", "No"])
    response_rate = st.slider("Host Response Rate (%)", 0, 100, 90)

# Calculate distance to center
manhattan_center = (40.7580, -73.9855)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

dist_to_center = haversine_distance(lat, lon, manhattan_center[0], manhattan_center[1])

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        try:
            # Create input dataframe with zeros
            input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
            
            # Fill in basic numeric values
            basic_features = {
                'accommodates': accommodates,
                'bedrooms': bedrooms,
                'beds': beds,
                'bathrooms': bathrooms,
                'minimum_nights': minimum_nights,
                'availability_30': availability_30,
                'availability_60': availability_30 * 2,  # approximate
                'availability_90': availability_30 * 3,  # approximate
                'availability_365': availability_30 * 12,  # approximate
                'number_of_reviews': num_reviews,
                'review_scores_rating': review_score,
                'host_is_superhost': 1 if is_superhost == "Yes" else 0,
                'instant_bookable': 1 if instant_bookable == "Yes" else 0,
                'host_response_rate': response_rate / 100,
                'latitude': lat,
                'longitude': lon,
                'dist_to_center': dist_to_center,
            }
            
            # Only set features that exist in the model
            for key, value in basic_features.items():
                if key in input_data.columns:
                    input_data[key] = value
            
            # Room type one-hot encoding
            room_type_map = {
                'Entire home/apt': 'room_type_Entire home/apt' if 'room_type_Entire home/apt' in input_data.columns else None,
                'Private room': 'room_type_Private room' if 'room_type_Private room' in input_data.columns else None,
                'Shared room': 'room_type_Shared room' if 'room_type_Shared room' in input_data.columns else None,
                'Hotel room': 'room_type_Hotel room' if 'room_type_Hotel room' in input_data.columns else None,
            }
            
            selected_room = room_type_map.get(room_type)
            if selected_room:
                input_data[selected_room] = 1
            
            # Borough one-hot encoding
            borough_map = {
                'Manhattan': 'neighbourhood_group_cleansed_Manhattan',
                'Brooklyn': 'neighbourhood_group_cleansed_Brooklyn',
                'Queens': 'neighbourhood_group_cleansed_Queens',
                'Bronx': 'neighbourhood_group_cleansed_Bronx',
                'Staten Island': 'neighbourhood_group_cleansed_Staten Island',
            }
            
            selected_borough = borough_map.get(borough)
            if selected_borough and selected_borough in input_data.columns:
                input_data[selected_borough] = 1
            
            # Predict
            pred_log = model.predict(input_data)[0]
            pred_price = np.expm1(pred_log)
            
            # Ensure reasonable range
            if pred_price < 10:
                pred_price = 10
            elif pred_price > 5000:
                pred_price = 5000
            
            # Show result with nice styling
            st.markdown(f"""
            <div style="text-align: center; padding: 2.5rem; 
                        background: linear-gradient(135deg, #667eea, #764ba2); 
                        color: white; border-radius: 20px; margin: 1.5rem 0;
                        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
                <h3 style="margin: 0; font-size: 1.3rem; opacity: 0.9;">💰 Predicted Nightly Price</h3>
                <h1 style="font-size: 4.5rem; margin: 1rem 0; font-weight: bold;">${pred_price:.2f}</h1>
                <p style="margin: 0; opacity: 0.8; font-size: 1.1rem;">
                    {room_type} in {borough} | {accommodates} guests | {bedrooms} bedroom(s)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📍 Distance to Center", f"{dist_to_center:.1f} km")
            with col2:
                st.metric("👥 Accommodates", f"{accommodates} guests")
            with col3:
                weekly_price = pred_price * 7
                st.metric("📅 Weekly Estimate", f"${weekly_price:.2f}")
            
            st.success(f"✅ Prediction complete!")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Try adjusting the input values. Some features might be missing.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- **Entire home/apt** commands highest prices
- **Manhattan** is the most expensive borough
- More **accommodates** = higher price
- **Superhosts** can charge premium
- Higher **review scores** slightly increase price
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 Model Info")
st.sidebar.markdown(f"- **Model:** CatBoost Tuned")
st.sidebar.markdown(f"- **R² Score:** 0.7519")
st.sidebar.markdown(f"- **Features:** {len(feature_names)}")