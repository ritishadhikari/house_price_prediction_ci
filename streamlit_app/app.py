import streamlit as st
import requests
import json
import time
import os
import socket
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add title and description
st.title("House Price Prediction - v2")
st.markdown(
    """
        <p style="font-size:18px; color: gray;">
        A simple MLOPs demonstration project for real-time house price prediction
        </p>
    """,
    unsafe_allow_html=True
    )

col1, col2=st.columns(2, gap='large')

#Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"""
    <p><strong>Square Footage:</strong><span id='sqft-value'></span></p>
    """,
    unsafe_allow_html=True
    )
    sqft=st.slider("",500,5000,1500,50,label_visibility="collapsed", key="sqft")
    st.markdown(f"""
    <script>
    document.getElementById('sqft-value').innerText = '{sqft} sq ft';
    </script>""", 
    unsafe_allow_html=True)

    bed_col, bath_col=st.columns(2)
    with bed_col:
        st.markdown("""
        <p>
        <strong>
        Bedrooms
        </strong>
        </p>
        """,
        unsafe_allow_html=True)
        bedrooms=st.selectbox("",options=[1,2,3,4,5,6],index=2,label_visibility="collapsed")
    with bath_col:
        st.markdown("""
        <p>
        <strong>
        Bathrooms
        </strong>
        </p>
        """,
        unsafe_allow_html=True)
        bathrooms=st.selectbox("",options=[1,1.5,2,2.5,3,3.5,4],index=2,label_visibility="collapsed")

    # location dropdown
    st.markdown("""
    <p>
    <strong>
    Location
    </strong>
    </p>
    """,
    unsafe_allow_html=True                    
    )
    location=st.selectbox("", options=["Urban","Suburban","Rural","Waterfront", "Mountain"], index=1, label_visibility="collapsed")

    # Year Built
    st.markdown(f"""
    <p>
    <strong>
    Year Built:
    </strong>
    <span id='year-value'>
    </span>
    </p>
    """, unsafe_allow_html=True)
    year_built=st.slider("",1900,2025,2000,1,label_visibility="collapsed",key="year")
    st.markdown(f"""
    <script>
    document.getElementById('year-value').innerText = '{year_built}'
    </script>
    """, unsafe_allow_html=True)

    predict_button=st.button("Predict_Price", use_container_width=True)

with col2:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown('<h2>Prediction Results</h2>',unsafe_allow_html=True)
    if predict_button:
        with st.spinner("Calculating Prediction..."):
            api_data={
                "sqft":sqft,
                "bedrooms":bedrooms,
                "bathrooms":bathrooms,
                "location":location.lower(),
                "year_built":year_built,
                "condition":"Good"
            }
            try:
                api_endpoint=os.environ.get('API_URL',"http://fastapi_app:8000/")
                if api_endpoint is None:
                    st.write("Not able to fetch the API_URL")
                    print("Not able to fetch the API_URL")
                # print(api_endpoint)
                predict_url=f"{api_endpoint.rstrip('/')}/predict"
                st.write(f"Connecting to API at: {predict_url}")
                response=requests.post(predict_url,json=api_data)
                response.raise_for_status()  # response exception for bad status code
                prediction=response.json()
                
                # Store prediction in session state
                st.session_state.prediction=prediction
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning("Using mock data for demonstration purposes. Please check your API connection.")
                st.session_state.prediction={
                    "predicted_price": 467145,
                    "confidence_interval": [420430.5, 513859.5],
                    "features_importance": {
                        "sqft": 0.43,
                        "location": 0.27,
                        "bathrooms": 0.15
                    },
                    "prediction_time": "0.12 seconds"
                }
                st.session_state.prediction_time=time.time()

            # display prediction if available
        if 'prediction' in st.session_state:
            pred=st.session_state.prediction
            formatted_price="${:,.0f}".format(pred['predicted_price'])
            st.markdown(f"<div class='prediction-value'>{formatted_price} </div>",unsafe_allow_html=True)

            # Display Confidence Score and Model Used
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown('<p class="info-label">Confidence Score</p>', unsafe_allow_html=True)
                st.markdown('<p class="info-value">92%</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            with col_b:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown('<p class="info-label">Model Used</p>', unsafe_allow_html=True)
                st.markdown('<p class="info-value">XGBoost</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                    
                
                # Display price range and prediction time
                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
                    lower = "${:,.1f}".format(pred["confidence_interval"][0])
                    upper = "${:,.1f}".format(pred["confidence_interval"][1])
                    st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
                with col_d:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
                    st.markdown('<p class="info-value">0.12 seconds</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Top factors
                st.markdown('<div class="top-factors">', unsafe_allow_html=True)
                st.markdown("<p><strong>Top Factors Affecting Price:</strong></p>", unsafe_allow_html=True)
                st.markdown("""
                <ul>
                    <li>Square Footage</li>
                    <li>Number of Bedrooms/Bathrooms</li>
                </ul>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown(
                """
                <div style="display: flex; height: 300px; allign-items: center;justify-content: center; color: #6b7280; text-align: center;">
                Fill out the form and click 'Predict Price' to see the estimated house price
                </div>
                """, unsafe_allow_html=True
            )
# Fetch version, hostname, and IP address
version = os.getenv("APP_VERSION", "1.0.0")  # Default version if not set in environment
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown(
    f"""
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Bootcamp</strong></p>
        <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Hostname:</strong> {hostname}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
                
