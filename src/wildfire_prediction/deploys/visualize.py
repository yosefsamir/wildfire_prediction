from pickle import NONE
import streamlit as st
import folium
from datetime import datetime
from streamlit_folium import st_folium
import geopandas as gpd
from .predict import predict
from .prepare_api_data import (
    prepare_data,
    fetch_weather_features
)
def store_prediction(prediction:float):
    """Store prediction data in session state."""
    # Check if point is inside California
    from shapely.geometry import Point
    point = Point(st.session_state.clicked['lon'], st.session_state.clicked['lat'])
    
    # Check if point is in California
    if not any(st.session_state.california.geometry.contains(point)):
        # Show error message without dismiss button
        st.error("Selected point is outside California!")
        # Clear prediction data
        st.session_state.prediction = ''
        if 'prediction_data' in st.session_state:
            del st.session_state.prediction_data
        return
    
    # If point is in California, generate prediction
    prob = prediction  # dummy probability
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state.prediction_data = {
        'probability': prob,
        'risk_level': "High" if prob > 0.66 else "Medium" if prob > 0.33 else "Low",
        'timestamp': now
    }


# ------- Helper Functions -------
def load_california_boundary(path: str = 'california.geojson') -> gpd.GeoDataFrame:
    """Load California GeoJSON boundary."""
    return gpd.read_file(path)


def initialize_session_state():
    """Initialize session state variables."""
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {'lat': None, 'lon': None, 'time': None}
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ''
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [20, 0]
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 2
    if 'zoom_to_california' not in st.session_state:
        st.session_state.zoom_to_california = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'show_error' not in st.session_state:
        st.session_state.show_error = False
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None


def create_map() -> folium.Map:
    """Create and return a Folium map based on session state."""
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        scrollWheelZoom=True
    )

    if st.session_state.zoom_to_california:
        inject_flyto_script(m)
        st.session_state.zoom_to_california = False

    if st.session_state.map_zoom >= 5:
        folium.GeoJson(
            st.session_state.california,
            name='California Boundary',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#ff7800',
                'weight': 2,
                'fillOpacity': 0,
                'interactive': False
            }
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def inject_flyto_script(m: folium.Map):
    """Add smooth fly-to California animation after zoom flag."""
    script = """
    <script>
    window.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            try {
                var iframes = document.querySelectorAll('iframe');
                for (var i = 0; i < iframes.length; i++) {
                    var iframe = iframes[i];
                    var doc = iframe.contentDocument || iframe.contentWindow.document;
                    var mapDiv = doc.querySelector('.leaflet-container');
                    if (mapDiv && mapDiv._leaflet_map) {
                        mapDiv._leaflet_map.flyToBounds([
                            [32.5, -124.4],
                            [42.0, -114.1]
                        ], {animate: true, duration: 1.5});
                        break;
                    }
                }
            } catch (e) { console.log(e); }
        }, 300);
    });
    </script>
    """
    folium.Element(script).add_to(m)


def sidebar_controls():
    """Render sidebar controls and handle interactions."""
    st.sidebar.header("Controls")

    if st.sidebar.button("Zoom to California", key="fly_to_ca"):
        st.session_state.map_center = [37.5, -119.5]
        st.session_state.map_zoom = 6
        st.session_state.zoom_to_california = True
        st.rerun()

    if st.sidebar.button("Predict Fire Risk", key="predict"):
        if st.session_state.clicked['lat'] is not None:
            features = fetch_weather_features(st.session_state.clicked['lat'], st.session_state.clicked['lon'], datetime.now().strftime("%Y-%m-%d"))
            prepared_data = prepare_data(features)
            prediction = predict(prepared_data)
            store_prediction(prediction)
            st.session_state.prediction = "active"
        else:
            st.sidebar.warning("Please select a location first")

    if st.session_state.clicked['lat'] is not None:
        if st.sidebar.button("Clear Selected Point"):
            st.session_state.clicked = {'lat': None, 'lon': None, 'time': None}
            st.session_state.prediction = ''
            if 'prediction_data' in st.session_state:
                del st.session_state.prediction_data
            st.rerun()


def run_prediction():
    """Run fire risk prediction."""
    if not st.session_state.clicked['lat']:
        st.warning("Please select a location first")
        return
        
    # Generate prediction if not already stored
    if 'prediction_data' not in st.session_state or not st.session_state.prediction_data:
        features = fetch_weather_features(st.session_state.clicked['lat'], st.session_state.clicked['lon'], datetime.now().strftime("%Y-%m-%d"))
        prepared_data = prepare_data(features)
        prediction = predict(prepared_data)
        store_prediction(prediction)
        if 'prediction_data' not in st.session_state or not st.session_state.prediction_data:  # If still None after store_prediction (e.g., outside California)
            return
    
    data = st.session_state.prediction_data
    risk_color = {
        "High": "#ff4b4b",
        "Medium": "#ffa641",
        "Low": "#37c463"
    }[data['risk_level']]

    # Card container
    st.markdown(
        f'<div style="background: linear-gradient(145deg, #fff, #f8f8f8); border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 5px solid {risk_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', 
        unsafe_allow_html=True
    )
    
    # Header and risk level
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div style="color: #2c3e50; font-size: 18px; font-weight: 600;">Fire Risk Assessment</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div style="background-color: {risk_color}; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px; font-weight: 500; text-align: center;">{data["risk_level"]} Risk</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('<div style="border-bottom: 2px solid #f0f0f0; margin: 10px 0;"></div>', unsafe_allow_html=True)
    
    # Probability display
    st.markdown(
        f'''
        <div style="background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; margin: 15px 0; text-align: center;">
            <div style="font-size: 24px; font-weight: bold; color: {risk_color};">{float(data["probability"]):.1%}</div>
            <div style="color: #666; font-size: 13px;">Probability of Fire Risk</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Location info
    st.markdown(
        f'''
        <div style="background: rgba(255,255,255,0.7); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <div style="color: #888; margin-bottom: 5px;">Location Details:</div>
            <div style="font-size: 14px; color: #444;">
                <b>Lat:</b> {st.session_state.clicked["lat"]:.6f}
                <b style="margin-left: 15px;">Lon:</b> {st.session_state.clicked["lon"]:.6f}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Time info
    st.markdown(
        f'''
        <div style="background: rgba(255,255,255,0.7); padding: 12px; border-radius: 8px;">
            <div style="color: #888; margin-bottom: 5px;">Assessment Time:</div>
            <div style="color: #444; font-size: 14px;">{data["timestamp"]}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Update status
    st.markdown(
        '<div style="margin-top: 15px; font-size: 12px; color: #999; text-align: right;">Updated just now</div>',
        unsafe_allow_html=True
    )
    
    # Close container
    st.markdown('</div>', unsafe_allow_html=True)


def display_info_and_prediction():
    """Display selected location info and prediction."""
    st.subheader("Location Info")
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

    st.subheader("Prediction")
    if st.session_state.prediction == "active":
        run_prediction()
    else:
        st.write("Make a prediction by selecting a point and clicking 'Predict Fire Risk'")


def handle_map_clicks(map_data):
    """Update session state with click events and map center/zoom."""
    if not map_data or not map_data.get('last_clicked'):
        return
        
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


def direct_coordinate_input():
    """Allow user to set coordinates manually."""
    st.subheader("Select Location")
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=37.5, min_value=-90.0, max_value=90.0, format="%.6f")
    lon = col2.number_input("Longitude", value=-119.5, min_value=-180.0, max_value=180.0, format="%.6f")
    
    if st.button("Set Location", key="set_loc", use_container_width=True):
        # Check if point is inside California before setting
        from shapely.geometry import Point
        point = Point(lon, lat)
        if not any(st.session_state.california.geometry.contains(point)):
            st.error("Selected point is outside California!")
            return
        
        st.session_state.clicked = {
            'lat': lat,
            'lon': lon,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.rerun()


def create_styled_popup(lat: float, lon: float, time: str) -> str:
    """Create a styled popup for map markers."""
    return f"""
    <div style="font-family:Arial; padding:5px;">
        <h4 style="margin-bottom:5px;">Selected Location</h4>
        <b>Latitude:</b> {lat:.6f}<br>
        <b>Longitude:</b> {lon:.6f}<br>
        <b>Time:</b> {time}
    </div>
    """


def apply_page_styling():
    """Apply global page styling including dark mode support."""
    st.markdown("""
    <style>
        /* Map styling */
        .folium-map {
            border: 2px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Error popup styling */
        .stAlert {
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            z-index: 9999 !important;
            max-width: 350px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        
        /* Marker styling */
        .leaflet-marker-icon {
            filter: hue-rotate(120deg);
        }
        
        /* Prediction box styling */
        div[data-testid="stMarkdown"] div[style*="background: linear-gradient"] {
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        div[data-testid="stMarkdown"] div[style*="background: linear-gradient"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            div[style*="background: linear-gradient"] {
                background: linear-gradient(145deg, #2c3e50, #1a2634) !important;
                color: #fff !important;
            }
            div[style*="background: linear-gradient"] h4 {
                color: #fff !important;
            }
            div[style*="background: rgba(255,255,255,0.7)"] {
                background: rgba(255,255,255,0.1) !important;
                color: #fff !important;
            }
            div[style*="color: #666"] {
                color: #aaa !important;
            }
            div[style*="color: #888"] {
                color: #bbb !important;
            }
            div[style*="color: #999"] {
                color: #888 !important;
            }
            div[style*="color: #444"] {
                color: #ddd !important;
            }
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            background-color: #ff7800;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #e65c00;
            box-shadow: 0 4px 12px rgba(230, 92, 0, 0.3);
            transform: translateY(-2px);
        }
        
        /* Improve overall text styling */
        .stMarkdown {
            font-family: 'Arial', sans-serif;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
    """, unsafe_allow_html=True)
