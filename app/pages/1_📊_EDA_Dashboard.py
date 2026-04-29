import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("*Interactive visualizations of NYC Airbnb listings data*")

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

@st.cache_data
def load_data():
    data_path = PROJECT_ROOT / "data" / "processed" / "processed_listings.csv"
    df = pd.read_csv(data_path)
    
    # Reconstruct borough from one-hot encoded columns
    borough_cols = {
        'Manhattan': 'neighbourhood_group_cleansed_Manhattan',
        'Brooklyn': 'neighbourhood_group_cleansed_Brooklyn',
        'Queens': 'neighbourhood_group_cleansed_Queens',
        'Bronx': 'neighbourhood_group_cleansed_Bronx',
        'Staten Island': 'neighbourhood_group_cleansed_Staten Island'
    }
    
    for borough, col in borough_cols.items():
        if col in df.columns:
            df.loc[df[col] == 1, 'borough'] = borough
    
    # Reconstruct room type from one-hot encoded columns
    room_cols = {
        'Entire home/apt': 'room_type_Entire home/apt',
        'Private room': 'room_type_Private room',
        'Shared room': 'room_type_Shared room',
        'Hotel room': 'room_type_Hotel room'
    }
    
    # Check which room type columns exist
    available_room_cols = {k: v for k, v in room_cols.items() if v in df.columns}
    
    if available_room_cols:
        for room, col in available_room_cols.items():
            df.loc[df[col] == 1, 'room_type'] = room
    
    # If no room type column exists (Entire home/apt is reference category)
    if 'room_type' not in df.columns:
        df['room_type'] = 'Entire home/apt'  # default
    
    # For rows where none of the room type dummies are 1
    room_dummy_cols = [c for c in df.columns if c.startswith('room_type_')]
    if room_dummy_cols:
        no_room = df[room_dummy_cols].sum(axis=1) == 0
        df.loc[no_room, 'room_type'] = 'Entire home/apt'
    
    return df

with st.spinner("Loading data..."):
    try:
        df = load_data()
        st.success(f"✅ Loaded {df.shape[0]:,} listings with {df.shape[1]} features")
    except FileNotFoundError:
        st.error("❌ Processed data file not found! Run 02_feature_engineering.ipynb first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Borough filter
if 'borough' in df.columns:
    borough_options = sorted(df['borough'].dropna().unique())
    boroughs = st.sidebar.multiselect(
        "Select Boroughs",
        options=borough_options,
        default=borough_options[:3] if len(borough_options) >= 3 else borough_options
    )
else:
    boroughs = []
    st.sidebar.warning("Borough column not found")

# Room type filter
if 'room_type' in df.columns:
    room_options = sorted(df['room_type'].dropna().unique())
    room_types = st.sidebar.multiselect(
        "Select Room Types",
        options=room_options,
        default=room_options
    )
else:
    room_types = []
    st.sidebar.warning("Room type column not found")

# Apply filters
filtered_df = df.copy()
if boroughs and 'borough' in df.columns:
    filtered_df = filtered_df[filtered_df['borough'].isin(boroughs)]
if room_types and 'room_type' in df.columns:
    filtered_df = filtered_df[filtered_df['room_type'].isin(room_types)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **Showing:** {len(filtered_df):,} listings")
if 'price_clean' in filtered_df.columns:
    st.sidebar.markdown(f"💰 **Median Price:** ${filtered_df['price_clean'].median():.0f}")
    st.sidebar.markdown(f"💵 **Mean Price:** ${filtered_df['price_clean'].mean():.0f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💰 Price Analysis", "🗺️ Geographic", "📈 Correlations", "🔧 Amenities"])

with tab1:
    st.subheader("💰 Price Distribution by Room Type")
    
    if 'room_type' in filtered_df.columns and 'price_clean' in filtered_df.columns:
        fig = px.box(filtered_df, x='room_type', y='price_clean', color='room_type',
                     labels={'price_clean': 'Price ($)', 'room_type': 'Room Type'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Price stats table
        st.subheader("📋 Price Statistics by Room Type")
        room_stats = filtered_df.groupby('room_type')['price_clean'].agg(['count', 'mean', 'median']).round(1)
        room_stats.columns = ['Count', 'Mean ($)', 'Median ($)']
        st.dataframe(room_stats, use_container_width=True)
    else:
        st.warning("Room type or price column not available")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏙️ Price by Borough")
        if 'borough' in filtered_df.columns and 'price_clean' in filtered_df.columns:
            fig = px.box(filtered_df, x='borough', y='price_clean',
                         color='borough',
                         labels={'price_clean': 'Price ($)', 'borough': 'Borough'},
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Borough or price column not available")
    
    with col2:
        st.subheader("📊 Price Distribution")
        if 'price_clean' in filtered_df.columns:
            fig = px.histogram(filtered_df, x='price_clean', nbins=50,
                              labels={'price_clean': 'Price ($)'},
                              color_discrete_sequence=['#667eea'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗺️ Price by Location")
    
    if 'price_clean' in filtered_df.columns:
        # Scatter map using lat/lon
        sample_size = min(3000, len(filtered_df))
        map_data = filtered_df.sample(sample_size)
        
        fig = px.scatter(map_data, x='longitude', y='latitude', 
                        color='price_clean', size='price_clean',
                        hover_data=['borough', 'room_type'] if 'borough' in map_data.columns else ['room_type'],
                        labels={'price_clean': 'Price ($)'},
                        color_continuous_scale='Viridis',
                        title='NYC Listings - Colored by Price')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distance vs Price
        st.subheader("📍 Distance to Center vs Price")
        if 'dist_to_center' in filtered_df.columns:
            fig = px.scatter(filtered_df.sample(min(3000, len(filtered_df))), 
                           x='dist_to_center', y='price_clean',
                           color='borough' if 'borough' in filtered_df.columns else None,
                           opacity=0.6, trendline='ols',
                           labels={'dist_to_center': 'Distance to Manhattan Center (km)', 
                                  'price_clean': 'Price ($)'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 Feature Correlations")
    
    numeric_cols = ['price_clean', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
                    'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                    'amenities_count', 'dist_to_center']
    available_cols = [c for c in numeric_cols if c in filtered_df.columns]
    
    if len(available_cols) > 2:
        corr = filtered_df[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10)
        ))
        fig.update_layout(height=550, width=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with price
        st.subheader("🔝 Top Correlations with Price")
        price_corr = corr['price_clean'].drop('price_clean').sort_values(ascending=False)
        fig = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=price_corr.values, color_continuous_scale='RdBu_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation analysis")

with tab4:
    st.subheader("🔧 Amenities Analysis")
    
    if 'amenities_count' in filtered_df.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Amenities", f"{filtered_df['amenities_count'].mean():.1f}")
        with col2:
            st.metric("Max Amenities", f"{filtered_df['amenities_count'].max():.0f}")
        with col3:
            st.metric("Min Amenities", f"{filtered_df['amenities_count'].min():.0f}")
        
        st.markdown("---")
        
        # Amenities vs Price
        sample_size = min(3000, len(filtered_df))
        fig = px.scatter(filtered_df.sample(sample_size), 
                        x='amenities_count', y='price_clean',
                        opacity=0.5, trendline='ols',
                        labels={'amenities_count': 'Number of Amenities', 
                                'price_clean': 'Price ($)'},
                        color_discrete_sequence=['#667eea'])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Popular amenities bars
        amenity_cols = ['has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_heating',
                       'has_washer', 'has_dryer', 'has_tv', 'has_pool', 'has_gym', 
                       'has_elevator', 'has_free_parking']
        available_amenities = [c for c in amenity_cols if c in filtered_df.columns]
        
        if available_amenities:
            st.subheader("📊 Popular Amenities")
            amenity_pct = filtered_df[available_amenities].mean() * 100
            amenity_pct = amenity_pct.sort_values(ascending=True)
            
            fig = px.bar(x=amenity_pct.values, y=[a.replace('has_', '').replace('_', ' ').title() for a in amenity_pct.index],
                        orientation='h',
                        labels={'x': '% of Listings', 'y': 'Amenity'},
                        color=amenity_pct.values, color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Amenities data not available")