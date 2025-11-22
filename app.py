import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Supply Chain Oracle",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #EFF6FF, #EDE9FE);
    }
    .high-risk {
        background-color: #FEE2E2;
        border-left: 4px solid #DC2626;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .medium-risk {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .low-risk {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Check if models exist
if not Path('models').exists():
    st.error("❌ Models not found! Please run 'python train_model.py' first to train the models.")
    st.stop()

# Load models and metadata
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lgb_model = joblib.load('models/lgb_model.pkl')
    le_carrier = joblib.load('models/le_carrier.pkl')
    le_weather = joblib.load('models/le_weather.pkl')
    le_traffic = joblib.load('models/le_traffic.pkl')
    
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return xgb_model, lgb_model, le_carrier, le_weather, le_traffic, metadata

xgb_model, lgb_model, le_carrier, le_weather, le_traffic, metadata = load_models()

# Title
st.markdown(f"""
    <div style='background: linear-gradient(to right, #4F46E5, #7C3AED); padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>📦 Supply Chain Oracle</h1>
        <p style='color: #E0E7FF; margin: 5px 0 0 0;'>AI-Powered Shipment Delay Predictor</p>
        <p style='color: #C7D2FE; margin: 5px 0 0 0; font-size: 14px;'>
            Model Accuracy: {metadata['ensemble_accuracy']*100:.1f}% | 
            Trained on {metadata['total_samples']} shipments
        </p>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload New Shipment Data for Predictions",
    type=['csv'],
    help="Upload CSV file with shipment data to predict delays"
)

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_prediction_data(file):
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.strip()
        return df
    
    df = load_prediction_data(uploaded_file)
    
    # Preprocess and predict
    @st.cache_data
    def make_predictions(data):
        df_pred = data.copy()
        
        # Handle missing values
        numeric_cols = df_pred.select_dtypes(include=[np.number]).columns
        df_pred[numeric_cols] = df_pred[numeric_cols].fillna(df_pred[numeric_cols].median())
        
        categorical_cols = df_pred.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['shipment_id']:
                df_pred[col] = df_pred[col].fillna('Unknown')
        
        # Feature engineering
        df_pred['speed_ratio'] = df_pred['distance_km'] / (df_pred['scheduled_days'] + 1)
        
        # Encode features
        df_pred['carrier_encoded'] = le_carrier.transform(df_pred['carrier'])
        df_pred['weather_encoded'] = le_weather.transform(df_pred['weather'])
        df_pred['traffic_encoded'] = le_traffic.transform(df_pred['traffic'])
        
        # Prepare features
        X = df_pred[metadata['feature_cols']]
        
        # Make predictions
        xgb_pred_proba = xgb_model.predict_proba(X)[:, 1]
        lgb_pred_proba = lgb_model.predict_proba(X)[:, 1]
        ensemble_pred_proba = (xgb_pred_proba + lgb_pred_proba) / 2
        
        # Add predictions to dataframe
        df_pred['delay_probability'] = ensemble_pred_proba
        df_pred['predicted_delay_days'] = (ensemble_pred_proba * df_pred['scheduled_days'] * 0.4).round()
        df_pred['predicted_arrival'] = df_pred['scheduled_days'] + df_pred['predicted_delay_days']
        df_pred['risk_level'] = pd.cut(
            ensemble_pred_proba,
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return df_pred, X
    
    with st.spinner('🔮 Analyzing shipments with AI models...'):
        df_results, X_features = make_predictions(df)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total Shipments", len(df_results))
    
    with col2:
        high_risk_count = (df_results['risk_level'] == 'High').sum()
        st.metric("⚠️ High Risk", high_risk_count)
    
    with col3:
        medium_risk_count = (df_results['risk_level'] == 'Medium').sum()
        st.metric("⚡ Medium Risk", medium_risk_count)
    
    with col4:
        avg_delay_prob = df_results['delay_probability'].mean()
        st.metric("📊 Avg Delay Prob", f"{avg_delay_prob*100:.1f}%")
    
    st.markdown("---")
    
    # Risk distribution chart
    st.subheader("📊 Risk Distribution Overview")
    
    risk_counts = df_results['risk_level'].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Shipment Risk Levels',
        color=risk_counts.index,
        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#DC2626'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Sidebar for shipment selection
    with st.sidebar:
        st.header("🔍 Shipment Selection")
        
        # Filter by risk level
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
        
        df_filtered = df_results[df_results['risk_level'].isin(risk_filter)]
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            options=['Delay Probability', 'Distance', 'Scheduled Days'],
            index=0
        )
        
        sort_mapping = {
            'Delay Probability': 'delay_probability',
            'Distance': 'distance_km',
            'Scheduled Days': 'scheduled_days'
        }
        
        df_filtered = df_filtered.sort_values(sort_mapping[sort_by], ascending=False)
        
        # Select shipment
        if len(df_filtered) > 0:
            shipment_ids = df_filtered['shipment_id'].tolist()
            selected_shipment_id = st.selectbox(
                "Select Shipment",
                options=shipment_ids,
                format_func=lambda x: f"{x} - {df_filtered[df_filtered['shipment_id']==x]['risk_level'].values[0]} Risk"
            )
        else:
            st.warning("No shipments match the selected filters")
            selected_shipment_id = None
    
    # Two column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📋 Shipment List")
        
        # Display top shipments
        for idx, row in df_filtered.head(15).iterrows():
            risk_class = f"{row['risk_level'].lower()}-risk"
            
            st.markdown(f"""
                <div class='{risk_class}'>
                    <strong>{row['shipment_id']}</strong> - {row['risk_level']} Risk<br>
                    📍 {row['origin']} → {row['destination']}<br>
                    🚚 {row['carrier']} | 📏 {row['distance_km']:.0f} km<br>
                    🌤️ {row['weather']} | 🚦 {row['traffic']} Traffic<br>
                    ⚡ Delay Probability: <strong>{row['delay_probability']*100:.1f}%</strong>
                </div>
            """, unsafe_allow_html=True)
        
        # Download predictions
        st.markdown("---")
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions (CSV)",
            data=csv,
            file_name="shipment_predictions.csv",
            mime="text/csv"
        )
    
    with col_right:
        st.subheader("🔬 Detailed Analysis")
        
        if selected_shipment_id and len(df_filtered) > 0:
            selected_row = df_results[df_results['shipment_id'] == selected_shipment_id].iloc[0]
            selected_idx = df_results[df_results['shipment_id'] == selected_shipment_id].index[0]
            
            # Risk assessment
            risk_class = f"{selected_row['risk_level'].lower()}-risk"
            risk_emoji = {'High': '🚨', 'Medium': '⚠️', 'Low': '✅'}
            
            st.markdown(f"""
                <div class='{risk_class}'>
                    <h3>{risk_emoji[selected_row['risk_level']]} {selected_row['risk_level']} Risk Shipment</h3>
                    <p><strong>Delay Probability:</strong> {selected_row['delay_probability']*100:.1f}%</p>
                    <p><strong>Shipment ID:</strong> {selected_row['shipment_id']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Route information
            st.markdown("### 🗺️ Route Information")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Origin:** {selected_row['origin']}")
                st.write(f"**Carrier:** {selected_row['carrier']}")
                st.write(f"**Weather:** {selected_row['weather']}")
            with col_b:
                st.write(f"**Destination:** {selected_row['destination']}")
                st.write(f"**Distance:** {selected_row['distance_km']:.0f} km")
                st.write(f"**Traffic:** {selected_row['traffic']}")
            
            # Delivery forecast
            st.markdown("### 📅 Delivery Forecast")
            
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Scheduled Delivery", f"{selected_row['scheduled_days']:.0f} days")
            with col_d:
                delta_text = f"+{selected_row['predicted_delay_days']:.0f} days" if selected_row['predicted_delay_days'] > 0 else "On time"
                st.metric(
                    "Predicted Delivery", 
                    f"{selected_row['predicted_arrival']:.1f} days",
                    delta=delta_text,
                    delta_color="inverse"
                )
            
            # SHAP values explanation
            st.markdown("### 🎯 Delay Contributing Factors (SHAP Values)")
            
            with st.spinner('Calculating SHAP values...'):
                explainer = shap.TreeExplainer(xgb_model)
                X_selected = X_features.iloc[[selected_idx]]
                shap_values = explainer.shap_values(X_selected)
                
                feature_names = ['Distance', 'Scheduled Days', 'Carrier', 'Weather', 'Traffic', 'Speed Ratio']
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0],
                    'Impact': ['Increases Delay' if x > 0 else 'Reduces Delay' for x in shap_values[0]]
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                fig_shap = px.bar(
                    shap_df,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    color='SHAP Value',
                    color_continuous_scale=['#10B981', '#FCD34D', '#DC2626'],
                    title='Feature Impact on Delay Prediction',
                    labels={'SHAP Value': 'Impact on Delay'}
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 AI-Powered Recommendations")
            
            recommendations = []
            
            if selected_row['risk_level'] in ['High', 'Medium']:
                # Weather-based recommendations
                if selected_row['weather'] in ['Storm', 'Rain', 'Fog']:
                    recommendations.append({
                        'type': 'warning',
                        'icon': '⛈️',
                        'title': 'Weather Alert',
                        'message': f"{selected_row['weather']} conditions predicted. Consider postponing by 1-2 days or rerouting via clearer regions."
                    })
                
                # Traffic-based recommendations
                if selected_row['traffic'] == 'High':
                    recommendations.append({
                        'type': 'info',
                        'icon': '🚦',
                        'title': 'Traffic Optimization',
                        'message': 'High traffic expected. Schedule departure during off-peak hours (10 PM - 5 AM) to reduce delays.'
                    })
                
                # Carrier-based recommendations
                if selected_row['carrier'] in ['Quick Ship', 'Swift Transport']:
                    recommendations.append({
                        'type': 'error',
                        'icon': '🚚',
                        'title': 'Carrier Optimization',
                        'message': f'Historical data shows {selected_row["carrier"]} has lower on-time performance. Consider switching to Speed Express or Fast Cargo.'
                    })
                
                # Distance-based recommendations
                if selected_row['distance_km'] > 1000:
                    recommendations.append({
                        'type': 'warning',
                        'icon': '✈️',
                        'title': 'Long Distance Route',
                        'message': f'{selected_row["distance_km"]:.0f}km is a long distance. For critical shipments, consider air freight or express rail options.'
                    })
                
                # Speed ratio recommendations
                if selected_row['speed_ratio'] < 100:
                    recommendations.append({
                        'type': 'info',
                        'icon': '⏱️',
                        'title': 'Tight Schedule',
                        'message': 'Low speed ratio indicates a tight delivery schedule. Add 1-2 buffer days for contingencies.'
                    })
            
            else:
                recommendations.append({
                    'type': 'success',
                    'icon': '✅',
                    'title': 'Optimal Route',
                    'message': 'This shipment is on track. Current route, carrier, and timing are optimal for on-time delivery.'
                })
            
            # Display recommendations
            for rec in recommendations:
                if rec['type'] == 'error':
                    st.error(f"{rec['icon']} **{rec['title']}:** {rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"{rec['icon']} **{rec['title']}:** {rec['message']}")
                elif rec['type'] == 'info':
                    st.info(f"{rec['icon']} **{rec['title']}:** {rec['message']}")
                else:
                    st.success(f"{rec['icon']} **{rec['title']}:** {rec['message']}")

else:
    # Welcome screen
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>📦 Welcome to Supply Chain Oracle</h2>
            <p style='font-size: 18px; color: #6B7280;'>
                Upload your shipment CSV file to start predicting delays with AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **🤖 XGBoost Model**
        
        Accuracy: {metadata['xgb_accuracy']*100:.1f}%
        """)
    
    with col2:
        st.info(f"""
        **⚡ LightGBM Model**
        
        Accuracy: {metadata['lgb_accuracy']*100:.1f}%
        """)
    
    with col3:
        st.success(f"""
        **🎯 Ensemble Model**
        
        Accuracy: {metadata['ensemble_accuracy']*100:.1f}%
        """)
    
    st.markdown("---")
    
    st.info("""
        **📋 Expected CSV Format:**
        - `shipment_id` - Unique identifier
        - `origin` - Starting city
        - `destination` - End city
        - `distance_km` - Distance in kilometers
        - `carrier` - Shipping company (must be one of: """ + ", ".join(metadata['carriers']) + """)
        - `weather` - Weather conditions (must be one of: """ + ", ".join(metadata['weather_conditions']) + """)
        - `traffic` - Traffic level (must be one of: """ + ", ".join(metadata['traffic_levels']) + """)
        - `scheduled_days` - Planned delivery time
        - `actual_days` - Actual delivery time (optional, for training data)
    """)
