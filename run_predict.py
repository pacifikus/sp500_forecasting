import datetime
import os
from datetime import date

import luigi
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml

from src.get_data import get_ticker_list, get_tickers_data
from src.preprocessing import make_stationary_series, compute_lags
from src.tda import transform_data_to_tda

config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['predict']
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])


class PredictionTask(luigi.Task):
    def requires(self):
        return None

    def output(self):
        return None

    def run(self):
        result_path = 'data/prediction.csv'
        sp500_tickers = get_ticker_list(is_random=True, size=5)
        data = get_tickers_data(sp500_tickers, config['data']['period'], config['data']['info_keys'])

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

            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/rf-model-{key}/Staging"
            )
            X_test, y_test = transform_data_to_tda(X, y, artefacts_postfix=key)
            prediction = df.iloc[-2]['price'] + np.exp(model.predict(X_test))
            result[key] = prediction[0]

        today = date.today() + datetime.timedelta(days=1)
        today = today.strftime('%Y-%m-%d')

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            result = pd.DataFrame(result.items(), columns=['shortName', today])
            df = df.merge(result, how='inner', on='shortName')
            df.to_csv(result_path, index=False)
        else:
            pd.DataFrame(result.items(), columns=['shortName', today]).to_csv(result_path, index=False)


if __name__ == '__main__':
    luigi.build([PredictionTask()])
