import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import plotly.express as px

# Streamlit Title
st.title("Cryptocurrency Price Prediction")

# Base URL for CoinGecko API
COINGECKO_API = "https://api.coingecko.com/api/v3"

# Function to fetch available cryptocurrencies
def get_available_cryptos():
    try:
        response = requests.get(f"{COINGECKO_API}/coins/list")
        response.raise_for_status()
        coins = response.json()
        return {coin['id']: coin['name'] for coin in coins}
    except Exception as e:
        st.error(f"Error fetching cryptocurrency list: {e}")
        return {}

# Function to fetch historical data
def get_historical_data(crypto_id, vs_currency, days):
    try:
        url = f"{COINGECKO_API}/coins/{crypto_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('timestamp', inplace=True)
        return prices
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# Function to prepare data for LightGBM
def prepare_data(data, time_step=100):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        return X, y, scaler
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, None

# Function to predict future prices
def predict_future(model, data, scaler, time_step=100, steps=30):
    try:
        data = scaler.transform(data)
        future_inputs = data[-time_step:].reshape(1, -1)
        
        predictions = []
        for _ in range(steps):
            pred = model.predict(future_inputs)
            predictions.append(pred[0])
            future_inputs = np.roll(future_inputs, -1)  # Shift array left
            future_inputs[0, -1] = pred[0]  # Assign prediction to the last element
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        return predictions.flatten()
    except Exception as e:
        st.error(f"Error predicting future prices: {e}")
        return None

# Fetch available cryptocurrencies
cryptos = get_available_cryptos()

# Debugging: Display fetched options
st.write("Available cryptocurrencies fetched:", cryptos)

if cryptos:
    mode = st.selectbox("Select Mode", ["Historical Data", "Future Predictions"])
    crypto_id = st.selectbox("Select Cryptocurrency", list(cryptos.keys()), format_func=lambda x: cryptos[x])
    vs_currency = st.selectbox("Select Currency", ["usd", "eur", "gbp", "jpy", "inr"])
    days = st.slider("Select number of days for historical data", 30, 365, 180)

    if st.button("Show Predictions"):
        st.header(f"{cryptos[crypto_id]} ({vs_currency.upper()})")

        prices = get_historical_data(crypto_id, vs_currency, days)
        if not prices.empty:
            if mode == "Historical Data":
                fig = px.line(
                    prices, 
                    x=prices.index, y='price', 
                    labels={"index": "Date", "price": "Price"},
                    title=f"{cryptos[crypto_id]} Historical Prices"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig)

            elif mode == "Future Predictions":
                data = prices['price'].values.reshape(-1, 1)
                X, y, scaler = prepare_data(data)

                if X is not None and y is not None and scaler is not None:
                    model = LGBMRegressor(n_estimators=100, max_depth=10)

                    try:
                        with st.spinner("Training the model, please wait..."):
                            model.fit(X, y)

                        future_predictions = predict_future(model, data[-100:], scaler)

                        if future_predictions is not None:
                            future_dates = pd.date_range(
                                start=prices.index[-1], periods=len(future_predictions)+1, freq="D")[1:]
                            historical_prices = prices['price'].values.flatten()
                            combined_prices = np.concatenate((historical_prices, future_predictions))

                            combined_dates = pd.Index(list(prices.index) + list(future_dates))

                            fig = px.line(
                                x=combined_dates,
                                y=combined_prices,
                                labels={"x": "Date", "y": "Price"},
                                title=f"{cryptos[crypto_id]} Price Prediction"
                            )
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig)
                        else:
                            st.error("Future predictions returned None.")
                    except Exception as e:
                        st.error(f"Error during model training or prediction: {e}")
        else:
            st.error("No historical data available.")
else:
    st.error("No cryptocurrency options available.")
    st.write("Please ensure you have a stable internet connection and the data source is accessible.")
