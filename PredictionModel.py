import streamlit as st
import plotly.express as px
from Historic_Crypto import HistoricalData, Cryptocurrencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

# Set up Streamlit for user inputs
st.title('Cryptocurrency Price Prediction')

# Function to get all available cryptocurrencies and their pairs
def get_available_currencies():
    try:
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        crypto_options = sorted(set([pair.split('-')[0] for pair in all_cryptos_df['id']]))
        return crypto_options
    except Exception as e:
        st.error(f"Error fetching cryptocurrencies: {e}")
        return []

def fetch_data(pair, start_date, end_date):
    try:
        return HistoricalData(pair, 60*60*24, start_date.strftime('%Y-%m-%d-00-00'), end_date.strftime('%Y-%m-%d-00-00'), verbose=False).retrieve_data()
    except Exception:
        return pd.DataFrame()

def get_data(cryptos, currency):
    pair = f'{cryptos}-{currency}'
    try:
        all_cryptos_df = Cryptocurrencies().find_crypto_pairs()
        if pair not in all_cryptos_df['id'].values:
            return None, f"{pair} not found in available cryptocurrency pairs."

        coinprices = pd.DataFrame()
        start_date = date(2020, 1, 1)
        end_date = date.today()
        delta = timedelta(days=100)
        
        with ThreadPoolExecutor() as executor:
            future_to_date = {executor.submit(fetch_data, pair, start_date + timedelta(days=i*100), min(start_date + timedelta(days=(i+1)*100), end_date)): i for i in range((end_date - start_date).days // 100 + 1)}
            for future in future_to_date:
                tmp = future.result()
                if not tmp.empty and 'close' in tmp.columns:
                    coinprices = pd.concat([coinprices, tmp[['close']]])

        if coinprices.empty:
            return None, f"No data available for {pair} from {date(2020, 1, 1)} to {date.today()}"

        coinprices.index = pd.to_datetime(coinprices.index)
        coinprices = coinprices.ffill()
        
        return coinprices, None
    
    except Exception as e:
        return None, str(e)

# Function to prepare data for LightGBM
def prepare_data(data, time_step=180):
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

# Function to make future predictions
def predict_future(model, data, scaler, time_step=100, steps=180):
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

# Main process
crypto_options = get_available_currencies()

if crypto_options:
    mode = st.selectbox('Select Mode', ['Historical Data', 'Future Predictions'])

    st.header("Available Cryptocurrencies")

    cryptos = st.selectbox('Select Coin', crypto_options)
    currency = st.selectbox('Select Currency', ['EUR', 'USD', 'USDT', 'GBP', 'JPY', 'KRW'])

    if cryptos and currency and st.button('Show Predictions'):
        st.header(f'{cryptos}-{currency}')

        coinprices, error = get_data(cryptos, currency)
        if coinprices is not None:
            if mode == 'Historical Data':
                fig = px.line(
                    x=coinprices.index, y=coinprices['close'],
                    labels={"x": "Date", "y": "Price"},
                    title=f'{cryptos}-{currency} Historical Prices'
                )
                fig.update_layout(
                    template='plotly_dark',
                    xaxis=dict(
                        gridcolor='rgb(75, 75, 75)',
                        tickfont=dict(color='white'),
                        title=dict(text='Date', font=dict(color='white'))
                    ),
                    yaxis=dict(
                        gridcolor='rgb(75, 75, 75)',
                        tickfont=dict(color='white'),
                        title=dict(text='Price', font=dict(color='white'))
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig)

            elif mode == 'Future Predictions':
                data = coinprices['close'].values.reshape(-1, 1)
                st.write(f"Data Shape: {data.shape}")
                X, y, scaler = prepare_data(data)

                if X is not None and y is not None and scaler is not None:
                    st.write(f"Prepared Data Shapes - X: {X.shape}, y: {y.shape}")
                    model = LGBMRegressor(n_estimators=100, max_depth=10)

                    try:
                        with st.spinner('Training the model, please wait...'):
                            model.fit(X, y)

                        st.write("Model training completed.")
                        
                        future_predictions = predict_future(model, data[-100:], scaler)

                        if future_predictions is not None:
                            st.write("Future predictions completed.")
                            future_dates = pd.date_range(start=coinprices.index[-1], periods=len(future_predictions)+1, freq='D')[1:]
                            historical_prices = coinprices['close'].values.flatten()
                            combined_prices = np.concatenate((historical_prices, future_predictions))

                            combined_dates = pd.Index(list(coinprices.index) + list(future_dates))

                            fig = px.line(
                                x=combined_dates,
                                y=combined_prices,
                                labels={"x": "Date", "y": "Price"},
                                title=f'{cryptos}-{currency} Price Prediction'
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                xaxis=dict(
                                    gridcolor='rgb(75, 75, 75)',
                                    tickfont=dict(color='white'),
                                    title=dict(text='Date', font=dict(color='white'))
                                ),
                                yaxis=dict(
                                    gridcolor='rgb(75, 75, 75)',
                                    tickfont=dict(color='white'),
                                    title=dict(text='Price', font=dict(color='white'))
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig)
                        else:
                            st.error("Future predictions returned None.")
                    except Exception as e:
                        st.error(f"Error during model training or prediction: {e}")
        else:
            st.error(error)
else:
    st.error("No cryptocurrency options available.")
