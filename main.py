# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import argparse
from datetime import datetime, timedelta

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def prepare_data(data):
    data['Days'] = (data.index - data.index[0]).days
    X = data['Days'].values.reshape(-1, 1)
    y = data['Value'].values
    return X, y

def train_model(X, y, model_type):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'polynomial':
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        return model, poly
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    else:
        raise ValueError("Unsupported model type. Choose 'linear', 'polynomial', or 'decision_tree'.")

    model.fit(X, y)
    return model

def predict_future(model, X, model_type, future_days):
    if model_type == 'polynomial':
        poly = PolynomialFeatures(degree=3)
        future_X = poly.fit_transform(np.array([len(X) + i for i in range(1, future_days + 1)]).reshape(-1, 1))
        return model.predict(future_X)
    else:
        future_X = np.array([len(X) + i for i in range(1, future_days + 1)]).reshape(-1, 1)
        return model.predict(future_X)

def plot_results(data, predictions, future_dates, model_type):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Value'], label='Historical Data', marker='o')
    plt.plot(future_dates, predictions, label=f'Predicted Future Data ({model_type})', marker='x', color='orange')
    
    # Adding confidence intervals (basic example)
    lower_bound = predictions - (0.1 * predictions)  # 10% lower
    upper_bound = predictions + (0.1 * predictions)  # 10% upper
    plt.fill_between(future_dates, lower_bound, upper_bound, color='orange', alpha=0.2, label='Confidence Interval')

    plt.title('Future Data Prediction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(args):
    data = load_data(args.data_file)
    X, y = prepare_data(data)

    model = train_model(X, y, args.model)
    predictions = predict_future(model, X, args.model, args.horizon)

    # Create future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, args.horizon + 1)]

    plot_results(data, predictions, future_dates, args.model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict future values based on historical data.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the historical data CSV file.')
    parser.add_argument('--model', type=str, choices=['linear', 'polynomial', 'decision_tree'], default='linear', help='Type of regression model to use.')
    parser.add_argument('--horizon', type=int, default=30, help='Number of days to predict into the future.')

    args = parser.parse_args()
    main(args)
