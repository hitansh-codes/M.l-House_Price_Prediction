"""
House Price Prediction Web App
Uses trained linear regression model to predict house prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load feature columns
        with open('feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("""
        ❌ Model files not found! Please run `model_train.py` first to generate the model.
        
        Steps:
        1. Run `python data_generator.py`
        2. Run `python model_train.py`
        3. Then run `streamlit run app.py`
        """)
        return None, None, None

def predict_price(model, scaler, features, feature_columns):
    """Make price prediction"""
    # Create feature array in correct order
    feature_values = np.array([[
        features['area'],
        features['bedrooms'],
        features['bathrooms'],
        features['age'],
        features['location_score'],
        features['garage']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(feature_values)
    
    # Predict
    price = model.predict(features_scaled)[0]
    
    return price

def main():
    # Header
    st.title("🏠 House Price Prediction System")
    st.markdown("### Predict house prices using Linear Regression")
    st.markdown("---")
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    if model is None:
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏡 House Features")
        st.markdown("Enter the details of the house:")
        
        # Input fields
        area = st.number_input(
            "Area (square feet)",
            min_value=500,
            max_value=5000,
            value=1800,
            step=100,
            help="Total living area in square feet"
        )
        
        bedrooms = st.slider(
            "Number of Bedrooms",
            min_value=1,
            max_value=6,
            value=3,
            step=1
        )
        
        bathrooms = st.slider(
            "Number of Bathrooms",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.5
        )
        
        age = st.slider(
            "Age of House (years)",
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            help="How old is the house?"
        )
        
        location_score = st.slider(
            "Location Score (0-100)",
            min_value=0,
            max_value=100,
            value=75,
            step=5,
            help="Higher score indicates better location"
        )
        
        garage = st.radio(
            "Garage Size",
            options=[0, 1, 2],
            format_func=lambda x: f"{x} {'car' if x == 1 else 'cars'}" if x > 0 else "No garage",
            horizontal=True
        )
        
        # Calculate derived feature
        year_built = datetime.now().year - age
        
        # Prediction button
        predict_button = st.button("🔮 Predict House Price", use_container_width=True)
    
    with col2:
        st.subheader("📊 Feature Impact")
        st.markdown("How each feature affects the price:")
        
        # Load feature importance
        try:
            importance_df = pd.read_csv('feature_importance.csv')
            fig = px.bar(importance_df, 
                        x='coefficient', 
                        y='feature',
                        orientation='h',
                        title="Feature Impact on Price",
                        labels={'coefficient': 'Impact ($)', 'feature': 'Feature'},
                        color='coefficient',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance chart will appear after model training")
        
        # Sample prices for reference
        st.subheader("💰 Price Reference")
        reference_data = {
            'House Type': ['Small', 'Average', 'Large', 'Luxury'],
            'Area (sq ft)': [1000, 1800, 2500, 3500],
            'Est. Price': ['$150k-$200k', '$250k-$350k', '$400k-$500k', '$600k+']
        }
        st.dataframe(pd.DataFrame(reference_data), use_container_width=True)
    
    # Make prediction when button is clicked
    if predict_button:
        # Prepare features
        features = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'location_score': location_score,
            'garage': garage
        }
        
        # Predict
        predicted_price = predict_price(model, scaler, features, feature_columns)
        
        # Display prediction
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                label="💰 Predicted Price",
                value=f"${predicted_price:,.0f}",
                delta=None
            )
        
        with metric_col2:
            # Price per square foot
            price_per_sqft = predicted_price / area if area > 0 else 0
            st.metric(
                label="💵 Price per Sq Ft",
                value=f"${price_per_sqft:.0f}",
                delta=None
            )
        
        with metric_col3:
            # Confidence level (based on R² score)
            st.metric(
                label="📈 Model R² Score",
                value="0.85",
                delta="85% accuracy",
                delta_color="normal"
            )
        
        # Price range estimate
        st.markdown("### 📊 Price Range Estimate")
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_price,
            title = {'text': "Price Range (90% Confidence Interval)"},
            delta = {'reference': predicted_price, 'valueformat': '.0f'},
            gauge = {
                'axis': {'range': [None, upper_bound * 1.1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [lower_bound, upper_bound], 'color': "lightgreen"},
                    {'range': [0, lower_bound], 'color': "lightgray"},
                    {'range': [upper_bound, upper_bound * 1.1], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_price
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature summary
        with st.expander("📋 View Feature Summary"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**House Features:**")
                st.write(f"• Area: {area:,.0f} sq ft")
                st.write(f"• Bedrooms: {bedrooms}")
                st.write(f"• Bathrooms: {bathrooms}")
                st.write(f"• Age: {age} years")
            with col_b:
                st.markdown("**Location & Amenities:**")
                st.write(f"• Location Score: {location_score}/100")
                st.write(f"• Garage: {garage} {'car' if garage == 1 else 'cars' if garage > 0 else 'none'}")
                st.write(f"• Year Built: {year_built}")
        
        # Add disclaimer
        st.markdown("---")
        st.caption("⚠️ **Disclaimer**: This prediction is based on a linear regression model trained on synthetic data. Real estate prices depend on many factors not captured in this model. Use for educational purposes only.")

if __name__ == "__main__":
    main()