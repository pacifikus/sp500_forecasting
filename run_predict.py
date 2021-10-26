import os.path

import numpy as np
import yaml

from src.get_data import get_ticker_list, get_tickers_data
from src.preprocessing import make_stationary_series, compute_lags
from src.tda import transform_data_to_tda
from utils import load_pickle

config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['predict']


def get_data():
    sp500_tickers = get_ticker_list(is_random=True, size=5)
    data = get_tickers_data(sp500_tickers, config['data']['period'], config['data']['info_keys'])
    return data


if __name__ == '__main__':
    data = get_data()
    grouped = data.groupby('symbol')
    result = {}

    for key, group in grouped:
        df = group.drop(
            config['data']['info_keys'],
            axis=1
        ).dropna()

        df = make_stationary_series(df)
        df.loc[df.shape[0]] = [0, 0, 0]
        df = compute_lags(df).dropna()
        y = df['log_diff'].tail(5)
        X = df.drop(['log_diff', 'log', 'price'], axis=1).tail(5)

        model = load_pickle(f"data/models/rf/{key}/model.pkl")
        X_test, y_test = transform_data_to_tda(X, y, artefacts_postfix=key)
        prediction = df.iloc[-2]['price'] + np.exp(model.predict(X_test))
        result[key] = prediction[0]
