import yfinance as yf
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si


np.random.seed(42)


def get_ticker_list(is_random=False, size=50):
    sp500_tickers = si.tickers_sp500()
    if is_random:
        return np.random.choice(sp500_tickers, size=size)
    return sp500_tickers


def get_info(ticker, period, info_keys):
    info_values = []
    ticker_obj = yf.Ticker(ticker)
    close_price = ticker_obj.history(period=period)['Close'].to_dict()

    for key in info_keys:
        info_values.append(ticker_obj.info.get(key, None))
    info = dict(zip(info_keys, info_values))
    info.update(close_price)
    return info


def get_tickers_data(tickers, period, info_keys):
    data = []

    for ticker in tickers:
        try:
            ticker_info = get_info(ticker, period, info_keys)
            data.append(ticker_info)
        except:
            continue

    data = pd.DataFrame(data)
    data.dropna(axis=1, how='all', inplace=True)
    return data
