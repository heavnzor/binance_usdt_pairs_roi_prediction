import pandas as pd
import numpy as np
import requests
import argparse
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from config import api_key
from dask import delayed, compute


def get_all_usdt_pairs(api_key, n=150):
    headers = {
        'X-MBX-APIKEY': api_key
    }
    response = requests.get(
        'https://api.binance.com/api/v3/ticker/24hr', headers=headers)
    if response.status_code == 200:
        data = response.json()
        usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
        sorted_pairs = sorted(
            usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
        return sorted_pairs[:n]
    else:
        raise Exception(
            f"Error retrieving USDT pairs: {response.text}")


def arima_model(data, order=(1, 0, 0)):
    model = sm.tsa.ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit


def predict_arima(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast[0]


def get_historical_data(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        historical_data = [{"time": item[0], "open": float(item[1]), "high": float(
            item[2]), "low": float(item[3]), "close": float(item[4])} for item in data]
        return historical_data
    else:
        raise Exception(
            f"Error retrieving historical data for {symbol}: {response.text}")


def preprocess_data(historical_data):
    close_prices = np.array([item['close']
                             for item in historical_data]).reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(close_prices)
    return normalized_data, scaler


def create_dataset(dataset, window_size=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - window_size):
        data_x.append(dataset[i:(i + window_size), 0])
        data_y.append(dataset[i + window_size, 0])
    return np.array(data_x), np.array(data_y)


def lstm_model(input_shape, learning_rate=0.001, neurons=50, compile_kwargs=None):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    compile_kwargs = compile_kwargs or {}
    optimizer = Adam(learning_rate=learning_rate)
    if 'loss' in compile_kwargs:
        del compile_kwargs['loss']
    if 'optimizer' in compile_kwargs:
        del compile_kwargs['optimizer']
    model.compile(optimizer=optimizer, loss='mse', **compile_kwargs)
    return model


def optimize_lstm_hyperparameters(X_train, y_train):
    model = KerasRegressor(
        model=lstm_model,
        model__input_shape=(X_train.shape[1], 1),
        model__learning_rate=None,
        model__neurons=None,
        epochs=100)

    param_grid = {
        'model__learning_rate': [0.01, 0.001, 0.0001],
        'model__neurons': [30, 50, 100],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def calculate_roi(price_today, predicted_price):
    return (predicted_price - price_today) / price_today


def get_linear_regression_predictions(X_train, y_train, X_test):
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    return model_linear.predict(X_test)


def get_lstm_predictions(X_train, y_train, X_test, best_params):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model_lstm = lstm_model(input_shape=(X_train.shape[1], 1), learning_rate=best_params['model__learning_rate'],
                            neurons=best_params['model__neurons'])
    model_lstm.fit(X_train, y_train, epochs=100, verbose=0)
    return model_lstm.predict(X_test)


def combine_predictions(predictions_linear, predictions_lstm, predictions_arima, weight_linear=0.3, weight_lstm=0.3,
                        weight_arima=0.4):
    return (predictions_linear * weight_linear) + (predictions_lstm * weight_lstm) + (predictions_arima * weight_arima)


def analyze_coin(coin, interval='1M', limit=1400, window_size=10):
    historical_data = get_historical_data(coin, interval=interval, limit=limit)
    processed_data, scaler = preprocess_data(historical_data)
    if len(processed_data) < 2 * window_size:
        return None
    X, y = create_dataset(processed_data, window_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Linear regression predictions
    predictions_linear = get_linear_regression_predictions(
        X_train, y_train, X_test)

    # LSTM predictions
    best_params = optimize_lstm_hyperparameters(X_train, y_train)
    predictions_lstm = get_lstm_predictions(
        X_train, y_train, X_test, best_params)

    # ARIMA predictions
    model_fit = arima_model(processed_data, order=(1, 0, 0))
    predictions_arima = predict_arima(model_fit, steps=window_size)

    # Combine predictions
    combined_predictions = combine_predictions(
        predictions_linear, predictions_lstm, predictions_arima)

    price_today = scaler.inverse_transform(
        processed_data[-1].reshape(-1, 1))[0][0]
    predicted_price = scaler.inverse_transform(combined_predictions)[-1][0]
    roi = calculate_roi(price_today, predicted_price)

    return roi


def main(period, output_file):
    all_usdt_pairs = get_all_usdt_pairs(api_key)
    # Convert all_usdt_pairs to a tuple of tuples
    all_usdt_pairs = [(item['symbol'], item['quoteVolume'])
                      for item in all_usdt_pairs]
    sorted_pairs = sorted(all_usdt_pairs, key=lambda x: x[1], reverse=True)

    # Analyze each coin using Dask
    delayed_tasks = []
    for coin in sorted_pairs:
        delayed_tasks.append(delayed(analyze_coin)(coin[0], period))

    # Compute the ROI predictions using Dask
    roi_predictions = dict(zip(sorted_pairs, compute(*delayed_tasks)))

    # Sort the ROI predictions and print the top coins
    sorted_roi_predictions = sorted(
        (item for item in roi_predictions.items() if item[1] is not None),
        key=lambda x: x[1], reverse=True)[:50]

    if len(sorted_roi_predictions) > 0:
        print("\nTop 50 coins to buy with potential ROI in the next {}:".format(period))
        for coin, roi in sorted_roi_predictions:
            print(f"{coin[0]}: {roi:.2f}%")

        if output_file:
            data = [(coin[0], roi) for coin, roi in sorted_roi_predictions]
            df = pd.DataFrame(data, columns=['Symbol', 'ROI'])
            df.to_csv(output_file, index=False)
    else:
        print("No ROI predictions available.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predict ROI for USDT pairs on Binance')
    parser.add_argument('--period', type=str, default='1w',
                        help='Period of time to predict ROI for (1w, 2w, 1M, etc.)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Name of the output file to save the results')
    args = parser.parse_args()

    main(args.period, args.output_file)
