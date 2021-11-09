import os.path

import luigi
import mlflow
import pandas as pd
import yaml

from src.evaluation import evaluate_model
from src.get_data import get_ticker_list, get_tickers_data
from src.preprocessing import timeseries_train_test_split, make_stationary_series, compute_lags
from src.tda import train_rf_on_tda, transform_data_to_tda

config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['train']
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment("forecasting")


class GetDataTask(luigi.Task):
    def requires(self):
        return None

    def output(self):
        return None

    def run(self):
        sp500_tickers = get_ticker_list(is_random=True, size=5)
        data = get_tickers_data(sp500_tickers, config['data']['period'], config['data']['info_keys'])
        data.to_csv(config['data']['data_path'], index=False)


class TrainingTask(luigi.Task):
    def requires(self):
        return None

    def output(self):
        return None

    def run(self):
        data = pd.read_csv(config['data']['data_path'])
        grouped = data.groupby('symbol')
        for key, group in grouped:
            df = group.drop(
                config['data']['info_keys'],
                axis=1
            ).dropna()

            df = make_stationary_series(df)
            df = compute_lags(df).dropna()
            y = df['log_diff']
            X = df.drop(['log_diff', 'log', 'price'], axis=1)
            X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

            with mlflow.start_run():
                model = train_rf_on_tda(X_train, y_train, artefacts_postfix=key)
                X_test, y_test = transform_data_to_tda(X_test, y_test, artefacts_postfix=key)
                mape = evaluate_model(model, X_train, y_train, X_test, y_test, postfix=key, need_plot=True)

                mlflow.log_param('window_size', config['tda']['sliding']['window_size'])
                mlflow.log_param('stride', config['tda']['sliding']['stride'])
                mlflow.log_metric("mape", mape)

                mlflow.sklearn.log_model(sk_model=model, artifact_path=f"rf-model-{key}",
                                         registered_model_name=f"rf-model-{key}")


if __name__ == '__main__':
    luigi.build([TrainingTask()])
