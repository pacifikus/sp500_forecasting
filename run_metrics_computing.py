import os.path
import yaml
import luigi
import pandas as pd
from src.monitoring import send_metrics

config_path = os.path.join('config/config.yaml')
config = yaml.safe_load(open(config_path))['train']


class ComputeMetricsTask(luigi.Task):
    def requires(self):
        return None

    def output(self):
        return None

    def run(self):
        real = pd.read_csv(
            config['data']['data_path'],
            index_col='symbol'
        ).iloc[:, -1]

        preds = pd.read_csv(
            config['data']['prediction_path'],
            index_col='shortName'
        ).iloc[:, -2]

        metrics = {}
        for item in real.index:
            metrics[item] = abs(real[item] - preds[item]) / abs(real[item]) * 100
        send_metrics(metrics)


if __name__ == '__main__':
    luigi.build([ComputeMetricsTask()])
