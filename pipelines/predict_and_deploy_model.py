"""Prediction pipeline for XGBoost wildfire prediction model.

This script uses a trained XGBoost model to make predictions on new data.
"""

import os
import json
import yaml
import argparse
import pandas as pd
import numpy as np
import logging
import streamlit as st
import folium
from datetime import datetime
from streamlit_folium import st_folium

import sys
# Add the src directory to the path so we can import our package
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from wildfire_prediction.deploys import (
    predict,
    prepare_data,
    apply_page_styling,
    create_map,
    create_styled_popup,
    direct_coordinate_input,
    display_info_and_prediction,
    handle_map_clicks,
    initialize_session_state,
    inject_flyto_script,
    run_prediction,
    load_california_boundary,
    sidebar_controls,
    store_prediction,
    st_folium,
    fetch_weather_features
)


# def prepare_prediction_data(data_path, config):
#     """Prepare data for prediction.
    
#     Args:
#         data_path: Path to data for prediction
#         config: Configuration dictionary
        
#     Returns:
#         tuple: (X, df) feature matrix and original dataframe
#     """
#     # Load data
#     logger.info(f"Loading data from {data_path}")
#     df = pd.read_csv(data_path)
    
#     logger.info(f"Data loaded with shape: {df.shape}")
    
#     # # Apply feature engineering if needed
#     # if 'grid_id' not in df.columns and all(col in df.columns for col in ['latitude', 'longitude']):
#     #     logger.info("Creating grid and time features...")
#     #     grid_size_km = config['feature_engineering']['grid_size_km']
#     #     df = create_grid_and_time_features(df, grid_size_km=grid_size_km)
    
#     # # Apply transformations to numerical features
#     # if 'frp_log' not in df.columns and 'frp' in df.columns:
#     #     logger.info("Transforming numerical features...")
#     #     log_transform_frp = config['feature_engineering']['log_transform_frp']
#     #     normalize_brightness = config['feature_engineering']['normalize_brightness']
#     #     df = transform_numerical_features(df, log_transform_frp=log_transform_frp, normalize_brightness=normalize_brightness)
    
#     # Get feature columns from config
#     feature_columns = config['feature_engineering'].get('feature_columns', [])
    
#     # If no specific features are defined, use all available numerical features
#     if not feature_columns:
#         # Exclude non-feature columns
#         exclude_cols = ['fire', 'acq_date', 'latitude', 'longitude']
#         feature_columns = [col for col in df.columns if col not in exclude_cols]
    
#     logger.info(f"Using features: {feature_columns}")
    
#     # Check if all feature columns exist in the dataframe
#     missing_cols = [col for col in feature_columns if col not in df.columns]
#     if missing_cols:
#         logger.warning(f"Missing feature columns: {missing_cols}")
#         # Remove missing columns from feature list
#         feature_columns = [col for col in feature_columns if col in df.columns]
    
#     # Create feature matrix
#     X = df[feature_columns]
    
#     logger.info(f"Prepared features with shape: {X.shape}")
    
#     return X, df


def main():
    # """Main function to make predictions using the trained XGBoost model."""
    # parser = argparse.ArgumentParser(description='Make predictions with XGBoost wildfire model')
    # parser.add_argument('--config', type=str, default='configs/params.yml',
    #                     help='Path to configuration file')
    # parser.add_argument('--model', type=str, 
    #                     default='artifacts/models/xgb_wildfire_model.pkl',
    #                     help='Path to trained model')
    # parser.add_argument('--data', type=str, required=True,
    #                     help='Path to data for prediction')
    # parser.add_argument('--output',defualt= 'artifacts/predictions', type=str, required=True,
    #                     help='Path to save prediction results')
    # args = parser.parse_args()
    
    st.title("Interactive Fire Map")
    apply_page_styling()

    initialize_session_state()
    st.session_state.california = load_california_boundary()
    
    # Display error popup if there's an error message
    if 'error_message' in st.session_state and st.session_state.error_message:
        with st.container():
            st.error(st.session_state.error_message)
            if st.button("Dismiss Error"):
                st.session_state.error_message = None
                st.rerun()
    
    # Sidebar controls
    sidebar_controls()
    
    # Create map
    m = create_map()
    
    # Add existing marker if any
    if st.session_state.clicked['lat'] is not None:
        popup_content = create_styled_popup(
            st.session_state.clicked['lat'],
            st.session_state.clicked['lon'],
            st.session_state.clicked['time']
        )
        folium.Marker(
            location=[st.session_state.clicked['lat'], st.session_state.clicked['lon']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color="green", icon="fire", prefix='fa'),
            tooltip="Click for details"
        ).add_to(m)

    # Display map and handle interactions
    map_data = st_folium(
        m,
        width=800,
        height=500,
        key="main_map"
    )
    
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.session_state.map_center = [map_data['center']['lat'], map_data['center']['lng']]
        st.session_state.map_zoom = map_data['zoom']
        st.session_state.clicked = {
            'lat': lat,
            'lon': lon,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Clear prediction when location changes
        st.session_state.prediction = ''
        if 'prediction_data' in st.session_state:
            del st.session_state.prediction_data
        st.rerun()

    # Location info and prediction below map
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Location Info")
        if st.session_state.clicked['lat'] is not None:
            st.markdown(
                f"""
                **Selected Location**  
                - **Latitude:** {st.session_state.clicked['lat']:.6f}  
                - **Longitude:** {st.session_state.clicked['lon']:.6f}  
                - **Selected at:** {st.session_state.clicked['time']}
                """
            )
        else:
            st.write("Click on the map to select a point...")

    with col2:
        st.markdown("### Prediction")
        if st.session_state.prediction == "active" and st.session_state.clicked['lat'] is not None:
            features = fetch_weather_features(st.session_state.clicked['lat'], st.session_state.clicked['lon'], datetime.now().strftime("%Y-%m-%d"))
            prepared_data = prepare_data(features)
            prediction = predict(prepared_data)
            store_prediction(prediction)
            run_prediction()
        else:
            st.write("Make a prediction by selecting a point and clicking 'Predict Fire Risk'")

    # Manual coordinate input at bottom
    st.markdown("---")
    direct_coordinate_input()
    
    st.markdown(
        """
        ### Instructions
        1. Click "Zoom to California" to focus map
        2. Click or input coordinates to select a point
        3. Click "Predict Fire Risk" to predict fire probablility
        4. Use "Clear Selected Point" to reset
        
        > **Tip:** You can also click directly on the map to select a location!
        """,
        unsafe_allow_html=True
    )
    
    
    # Print summary
    logger.info(f"Prediction summary:")
    logger.info(f"Prediction pipeline completed successfully!")


if __name__ == '__main__':
    main()