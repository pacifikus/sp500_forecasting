import os
import yaml
from src.get_data import get_ticker_list, get_tickers_data
from src.preprocessing import timeseries_train_test_split, make_stationary_series, compute_lags
from src.evaluation import evaluate_model
from src.tda import train_rf_on_tda, transform_data_to_tda
import mlflow
import os.path
import pandas as pd
import uuid


config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['train']


def get_data(load_data=False):
    if not load_data and os.path.exists('data/raw/data.csv'):
        return pd.read_csv('data/raw/data.csv')
    sp500_tickers = get_ticker_list(is_random=True, size=5)
    data = get_tickers_data(sp500_tickers, config['data']['period'], config['data']['info_keys'])
    data.to_csv('data/raw/data.csv', index=False)
    return data


if __name__ == '__main__':
    data = get_data()
    grouped = data.groupby('symbol')
    for key, group in grouped:
        df = group.drop(
            ['symbol', 'shortName', 'sector', 'industry', 'country'],
            axis=1
        ).dropna()

        df = make_stationary_series(df)
        df = compute_lags(df).dropna()
        y = df['log_diff']
        X = df.drop(['log_diff', 'log', 'price'], axis=1)
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

        with mlflow.start_run():
            params = {
                'window_size': config['tda']['sliding']['window_size'],
                'stride': config['tda']['sliding']['stride'],
                'dimension': config['tda']['embedder']['dimension'],
                'delay': config['tda']['embedder']['delay'],
                'seed': config['seed'],
            }
            model = train_rf_on_tda(X_train, y_train, params, artefacts_postfix=key)
            X_test, y_test = transform_data_to_tda(X_test, y_test, artefacts_postfix=key)
            mape = evaluate_model(model, X_train, y_train, X_test, y_test, postfix=key, need_plot=True)

            mlflow.log_param('window_size', params['window_size'])
            mlflow.log_param('stride', params['stride'])
            mlflow.log_metric("mape", mape)
            mlflow.sklearn.save_model(model, f"data/models/rf/{key}_{uuid.uuid4()}")

            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
