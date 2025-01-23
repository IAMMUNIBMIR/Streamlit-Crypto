import streamlit as st
import plotly.express as px
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from datetime import date, timedelta

# Set up Streamlit for user inputs
st.title('Cryptocurrency Price Prediction')

# Initialize the CoinGecko API client
cg = CoinGeckoAPI()

# Function to get all available cryptocurrencies from CoinGecko and clean the list
def get_available_currencies():
    try:
        # Fetch the list of coins from CoinGecko
        coins = cg.get_coins_list()
        
        # Mapping cryptocurrency symbols to CoinGecko's internal IDs
        symbol_to_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'LTC': 'litecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'SOL': 'solana',
            'LINK': 'chainlink',
            'DOT': 'polkadot',
            'UNI': 'uniswap'
        }
        
        # Extract the list of main cryptocurrencies (without duplicates)
        crypto_options = list(symbol_to_id.keys())
        
        return sorted(crypto_options), symbol_to_id
    except Exception as e:
        st.error(f"Error fetching cryptocurrencies: {e}")
        return [], {}

def fetch_data(crypto_symbol, start_date, end_date, symbol_to_id):
    try:
        # Use the symbol_to_id mapping to get the correct CoinGecko ID
        crypto_id = symbol_to_id.get(crypto_symbol)
        if not crypto_id:
            st.error(f"Invalid cryptocurrency symbol: {crypto_symbol}")
            return pd.DataFrame()

        # Convert the start and end dates to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        # Fetch historical data from CoinGecko
        data = cg.get_coin_market_chart_range_by_id(id=crypto_id, vs_currency='usd', from_timestamp=start_timestamp, to_timestamp=end_timestamp)
        
        # Convert the data into a pandas DataFrame
        coin_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        coin_prices['timestamp'] = pd.to_datetime(coin_prices['timestamp'], unit='ms')
        coin_prices.set_index('timestamp', inplace=True)
        return coin_prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

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

def predict_future(model, data, scaler, time_step=100, steps=120):
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
crypto_options, symbol_to_id = get_available_currencies()

if crypto_options:
    # Display the mode selectbox and other inputs if data was fetched
    mode = st.selectbox('Select Mode', ['Historical Data', 'Future Predictions'])
    st.header("Available Cryptocurrencies")

    cryptos = st.selectbox('Select Coin', crypto_options)
    currency = st.selectbox('Select Currency', ['USD', 'EUR', 'GBP', 'JPY', 'KRW'])

    if cryptos and currency and st.button('Show Predictions'):
        st.header(f'{cryptos}-{currency}')
        st.write(f"Fetching data for {cryptos}-{currency}...")

        # Fetch the data based on user selection
        coinprices = fetch_data(cryptos, date(2020, 1, 1), date.today(), symbol_to_id)
        
        if not coinprices.empty:
            if mode == 'Historical Data':
                try:
                    fig = px.line(
                        x=coinprices.index, y=coinprices['price'],
                        labels={"x": "Date", "y": "Price (USD)"},
                        title=f'{cryptos} Historical Prices'
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
                            title=dict(text='Price (USD)', font=dict(color='white'))
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error displaying historical data: {e}")

            elif mode == 'Future Predictions':
                try:
                    data = coinprices['price'].values.reshape(-1, 1)
                    X, y, scaler = prepare_data(data)

                    if X is not None and y is not None and scaler is not None:
                        model = LGBMRegressor(n_estimators=100, max_depth=10)

                        with st.spinner('Training the model, please wait...'):
                            model.fit(X, y)
                        
                        future_predictions = predict_future(model, data[-100:], scaler)

                        if future_predictions is not None:
                            future_dates = pd.date_range(start=coinprices.index[-1], periods=len(future_predictions)+1, freq='D')[1:]
                            historical_prices = coinprices['price'].values.flatten()
                            combined_prices = np.concatenate((historical_prices, future_predictions))

                            combined_dates = pd.Index(list(coinprices.index) + list(future_dates))

                            fig = px.line(
                                x=combined_dates,
                                y=combined_prices,
                                labels={"x": "Date", "y": "Price (USD)"},
                                title=f'{cryptos} Price Prediction'
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
                                    title=dict(text='Price (USD)', font=dict(color='white'))
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig)
                        else:
                            st.error("Future predictions returned None.")
                    else:
                        st.error("Data preparation failed.")
                except Exception as e:
                    st.error(f"Error during future prediction: {e}")
        else:
            st.error(f"Failed to fetch data for {cryptos}")
else:
    st.error("No cryptocurrency options available.")
