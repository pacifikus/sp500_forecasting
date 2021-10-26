import numpy as np
import statsmodels.api as sm


def make_stationary_series(series, target_col='price'):
    if series.shape[0] == 1:
        series = series.T
    series.columns = [target_col]
    series['log'] = np.log(series[target_col])
    series['log_diff'] = series['log'] - series['log'].shift(1)
    return series


def compute_lags(series, start=6, end=25):
    for i in range(start, end):
        series["lag_{}".format(i)] = series['log_diff'].shift(i)
    return series


def timeseries_train_test_split(X, y, test_size=0.3):
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


def make_stationary_df(data):
    return data.apply(lambda x: make_stationary_series(x), axis=0)
