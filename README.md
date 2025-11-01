# St. Louis Temperature Predictor
This repository contains a **machine learning pipeline** to predict the average daily temperature in St. Louis using historical weather data, along with a **Streamlit app** for interactive predictions.

## Repository Structure
stlouis-weather-predictor/

│

├── weather-prediction.ipynb # Jupyter notebook to preprocess data and train neural network

├── stlouis_weather.csv # Historical weather data (NOAA GHCN-Daily)

├── stlouis_temperature_predictor.keras # Trained Keras model

├── scaler.save # Saved MinMaxScaler for feature normalization

├── weather_app.py # Streamlit app to interact with the model

├── requirements.txt # Python dependencies

└── README.md # Project README

## Training Notebook (```weather-prediction.ipynb```)
The notebook contains:
1. **Data Loading**
   * Loads the NOAA GHCN-Daily CSV for St. Louis.
   * Extracts features: ```TMIN```, ```TMAX```, ```PRCP```, ```AWND```, and ```day_of_year```.
2. **Data Preprocessing**
   * Handles missing values.
   * Normalizes features using ```MinMaxScaler```.
3. **Neural Network Training**
   * Uses TensorFlow/Keras to build a simple feedforward network.
   * Architecture:
     * Dense(64, ReLU)
     * Dense(32, ReLU)
     * Dense(1) output for predicted average temperature
   * Loss: Mean Squared Error (MSE), Metric: Mean Absolute Error (MAE).
4. **Evaluation and Visualization**
   * Splits data into train/test sets.
   * Plots training and validation loss.
   * Compares predicted vs actual temperatures.
5. Model Saving
   * Saves trained model (```.keras```) and scaler (```.save```) for use in the Streamlit app.
  
## Streamlit App (```weather_app.py```)
The app allows users to **predict daily average temperature** interactively.
**Features**:
* Select a date (past or up to 7 days in the future).
* Fetches weather data from **Open-Meteo**.
* Shows **TMIN, TMAX, precipitation, wind speed** used for prediction.
* Predicts **average temperature** in Celsius and Fahrenheit.
* Uses **cached API calls** to reduce latency.

**Run this app:
```bash
streamlit run weather_app.py
```

## Requirements
* Python 3.9+
* Streamlit
* TensorFlow / Keras
* scikit-learn
* joblib
* requests
* numpy, pandas
Install dependencies with:
``` bash
pip install -r requirements.txt
```

## Usage Workflow
1. **Train the model (optional)**
   * Open ```weather-prediction.ipynb```
   * Run the notebook to train your own model and save it with the scaler.
2. **Run this app**
   * Ensure the trained model and scaler are in the same folder.
   * Start Streamlit: ```streamlit run weather_app.py```.
3. **Select a date
   * The app fetches weather features for the selected date.
   * Displays both fetched data and predicted average temperature.

## Notes
* The neural network predicts **average temperature only**.
* Historical data is used for past dates; forecast data is used for today/future dates.
* Cached API results expire after 1 hour.

## License
This project is licensed under the MIT License.

