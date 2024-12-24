import streamlit as st
import plotly.express as px
from Historic_Crypto import HistoricalData, Cryptocurrencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from datetime import date, timedelta

# Set up Streamlit app
st.title("Cryptocurrency Price Prediction")

# Fetch available cryptocurrencies
def get_available_currencies():
    try:
        cryptos = Cryptocurrencies().find_crypto_pairs()
        crypto_options = sorted(set([pair.split('-')[0] for pair in cryptos['id']]))
        return crypto_options
    except Exception as e:
        st.error(f"Error fetching cryptocurrencies: {e}")
        return []

# Fetch historical data in chunks
def fetch_historical_data(pair, start_date, end_date):
    try:
        coinprices = HistoricalData(
            pair,
            60 * 60 * 24,  # Daily interval
            start_date.strftime('%Y-%m-%d-00-00'),
            end_date.strftime('%Y-%m-%d-00-00'),
            verbose=False
        ).retrieve_data()
        return coinprices
    except Exception:
        return pd.DataFrame()

# Consolidate historical data
def get_data(pair):
    start_date = date(2020, 1, 1)
    end_date = date.today()
    coinprices = pd.DataFrame()
    delta = timedelta(days=100)

    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + delta, end_date)
        chunk = fetch_historical_data(pair, current_date, next_date)
        if not chunk.empty:
            coinprices = pd.concat([coinprices, chunk[['close']]])
        current_date = next_date

    if coinprices.empty:
        return None, f"No data found for {pair}."
    coinprices.index = pd.to_datetime(coinprices.index)
    return coinprices.ffill(), None

# Prepare data for LightGBM
def prepare_data(data, time_step=100):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i - time_step:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y), scaler
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, None

# Predict future prices
def predict_future(model, data, scaler, time_step=100, steps=120):
    try:
        scaled_data = scaler.transform(data)
        future_inputs = scaled_data[-time_step:].reshape(1, -1)

        predictions = []
        for _ in range(steps):
            pred = model.predict(future_inputs)
            predictions.append(pred[0])
            future_inputs = np.roll(future_inputs, -1)
            future_inputs[0, -1] = pred[0]

        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"Error predicting future prices: {e}")
        return None

# Main Streamlit app logic
crypto_options = get_available_currencies()
if crypto_options:
    mode = st.selectbox("Select Mode", ["Historical Data", "Future Predictions"])
    st.header("Available Cryptocurrencies")
    cryptos = st.selectbox("Select Coin", crypto_options)
    currency = st.selectbox("Select Currency", ["EUR", "USD", "USDT", "GBP", "JPY", "KRW"])

    if st.button("Show Predictions"):
        pair = f"{cryptos}-{currency}"
        st.header(f"{pair}")

        coinprices, error = get_data(pair)
        if coinprices is not None:
            if mode == "Historical Data":
                fig = px.line(
                    x=coinprices.index, y=coinprices["close"],
                    labels={"x": "Date", "y": "Price"},
                    title=f"{pair} Historical Prices"
                )
                st.plotly_chart(fig)

            elif mode == "Future Predictions":
                data = coinprices["close"].values.reshape(-1, 1)
                X, y, scaler = prepare_data(data)

                if X is not None and y is not None and scaler is not None:
                    model = LGBMRegressor(n_estimators=100, max_depth=10)
                    with st.spinner("Training the model..."):
                        model.fit(X, y)

                    future_predictions = predict_future(model, data[-100:], scaler)
                    if future_predictions is not None:
                        future_dates = pd.date_range(
                            start=coinprices.index[-1], periods=len(future_predictions) + 1, freq="D"
                        )[1:]
                        combined_prices = np.concatenate((coinprices["close"].values.flatten(), future_predictions))
                        combined_dates = list(coinprices.index) + list(future_dates)

                        fig = px.line(
                            x=combined_dates,
                            y=combined_prices,
                            labels={"x": "Date", "y": "Price"},
                            title=f"{pair} Price Prediction"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.error("Future predictions could not be generated.")
        else:
            st.error(error)
else:
    st.error("No cryptocurrency options available.")
