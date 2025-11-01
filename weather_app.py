import streamlit as st
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime
import requests
import time

# Constants
LAT = 38.6270 # St. Louis latitude
LON = -90.1994 # St. Louis longitude
MODEL_PATH = 'stlouis_temperature_predictor.keras'
SCALER_PATH = 'scaler.save'

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Helper: Fetch weather with retry
def fetch_weather_with_retry(url, retries=3, delay=5):
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to fetch weather data after {retries} attempts: {e}")

# Helper: Fetch weather data
def fetch_weather(lat, lon, date):
    '''
    Fetch TMIN, TMAX, precipitation and wind speed.
    Uses historical data for past dates and forcast for today/future dates.
    '''

    if isinstance(date, dict):
        date = datetime.date(date['year'], date['month'], date['day'])
    elif isinstance(date, datetime.datetime):
        date = date.date()

    today = datetime.date.today()

    if date <= today:
        # Historical data
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&timezone=America/Chicago"
        )
    else:
        # Forecast data
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
            f"precipitation_sum,windspeed_10m_max&timezone=America/Chicago"
            f"&start_date={date}&end_date={date}"
        )

    data = fetch_weather_with_retry(url)

    if date <= today:
        try:
            daily = data['daily']
            tmin = data['daily']['temperature_2m_min'][0]
            tmax = data['daily']['temperature_2m_max'][0]
            prcp = data['daily']['precipitation_sum'][0]
            wind = data['daily']['windspeed_10m_max'][0]
        except KeyError:
            raise RuntimeError("Unexpected historical data format from API")
    else:
        # Forecast API returns arrays for multiple days; find the index of the requested date
        try:
            daily = data['daily']
            date_index = daily['time'].index(date.isoformat())
            tmin = daily['temperature_2m_min'][date_index]
            tmax = daily['temperature_2m_max'][date_index]
            prcp = daily['precipitation_sum'][date_index]
            wind = daily['windspeed_10m_max'][date_index]
        except (KeyError, ValueError):
            raise RuntimeError(f"Forecast data not available for {date}")

    return tmin, tmax, prcp, wind

# Cached wrapper for Streamlit
@st.cache_data(ttl=3600) # cache results for 1 hour
def get_weather_data(date):
    return fetch_weather(LAT, LON, date)

# Streamlit UI
st.title("🌤️St. Louis Temperature Predictor")
st.markdown("""
This app predicts the **average daily temperature** in St. Louis
using historical weather data from **Open-Meteo** and your trained neural network.
""")

# Sidebar: Date Input
st.sidebar.header("Select Date")
selected_date = st.sidebar.date_input(
    "Choose a date",
    value=datetime.date.today(),
    min_value=datetime.date(2023, 1, 1),
    max_value=datetime.date.today() + datetime.timedelta(days=7) # allow 7 days ahead
)

# Fetch data
with st.spinner("Fetching weather data..."):
    try:
        tmin_c, tmax_c, prcp, awnd = get_weather_data(selected_date)
    except Exception as e:
        st.error(f"Could not fetch weather data: {e}")
        st.stop()

# Convert fetched temperature to Fahrenheit
tmin_f = tmin_c * 9/5 + 32
tmax_f = tmax_c * 9/5 + 32

# Display fetched weather info
st.subheader("Weather Data Used for Prediction")
st.write(f"**Date:** {selected_date.strftime('%B %d, %Y')}")
st.write(f"**Minimum Temperature:** {tmin_f:.2f} °F ({tmin_c:.2f} °C)")
st.write(f"**Maximum Temperature:** {tmax_f:.2f} °F ({tmax_c:.2f} °C)")
st.write(f"**Precipitation:** {prcp:.2f} mm")
st.write(f"**Wind Speed:** {awnd:.2f} m/s")

# Prepare feature for prediction
day_of_year = selected_date.timetuple().tm_yday
user_input = np.array([[tmin_c, tmax_c, prcp, awnd, day_of_year]], dtype='float32')

# Scale input
user_input_scaled = scaler.transform(user_input)

# Make predicition
predicted_temp_c = model.predict(user_input_scaled)[0][0]
predicted_temp_f = predicted_temp_c * 9/5 + 32

# Display Output
st.subheader("Predicted Average Temperature:")
st.write(f"{predicted_temp_f:.2f} °F ( {predicted_temp_c:.2f} °C)")

# Footer
st.caption("Data source: [Open-Meteo.com](https://open-meteo.com/) | Model trained on NOAA dataset")