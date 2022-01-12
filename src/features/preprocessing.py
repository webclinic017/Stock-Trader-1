from typing import Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators
import ta


def stock_preprocessing(df_data: pd.DataFrame, price_ref_col:str,
                        open_col:str = 'Open', close_col: str = 'Close',
                        min_col: str = 'Low', max_col: str = 'High', williams_r_lbp=14,
                        init_datetime=datetime(2000, 1, 1)) -> pd.DataFrame:

    # MACD, Signal, Histogram:
    df_data['macd'] = ta.trend.macd(df_data[close_col])
    df_data['signal'] = ta.trend.macd_signal(df_data[close_col])
    df_data['histogram'] = ta.trend.macd_diff(df_data[close_col])
    # William %R
    df_data['williams_r'] = ta.momentum.williams_r(df_data[max_col], df_data[min_col],
                                                   df_data[close_col], lbp=williams_r_lbp)
    # SMA
    df_data['sma9'] = ta.trend.sma_indicator(df_data[price_ref_col], window=9)
    df_data['sma21'] = ta.trend.sma_indicator(df_data[price_ref_col], window=21)
    df_data['sma50'] = ta.trend.sma_indicator(df_data[price_ref_col], window=50)
    # SMA-related features
    df_data['pref_sma9_perc'] = (df_data[price_ref_col] - df_data['sma9']) / df_data['sma9']
    df_data['pref_sma21_perc'] = (df_data[price_ref_col] - df_data['sma21']) / df_data['sma21']
    df_data['pref_sma50_perc'] = (df_data[price_ref_col] - df_data['sma50']) / df_data['sma50']
    df_data['sma9gt21'] = df_data['sma9'] > df_data['sma21']
    df_data['sma9gt50'] = df_data['sma9'] > df_data['sma50']
    df_data['sma21gt50'] = df_data['sma21'] > df_data['sma50']

    # EMA
    df_data['ema9'] = ta.trend.ema_indicator(df_data[price_ref_col], window=9)
    df_data['ema21'] = ta.trend.ema_indicator(df_data[price_ref_col], window=21)
    df_data['ema50'] = ta.trend.ema_indicator(df_data[price_ref_col], window=50)
    # EMA-related features
    df_data['pref_ema9_perc'] = (df_data[price_ref_col] - df_data['ema9']) / df_data['ema9']
    df_data['pref_ema21_perc'] = (df_data[price_ref_col] - df_data['ema21']) / df_data['ema21']
    df_data['pref_ema50_perc'] = (df_data[price_ref_col] - df_data['ema50']) / df_data['ema50']
    df_data['ema9gt21'] = df_data['ema9'] > df_data['ema21']
    df_data['ema9gt50'] = df_data['ema9'] > df_data['ema50']
    df_data['ema21gt50'] = df_data['ema21'] > df_data['ema50']

    # Daily spread:
    df_data['open_close_diff_perc']  = (df_data[close_col] - df_data[open_col])/df_data[open_col]
    df_data['open_close_diff_ratio']  = (df_data[close_col] - df_data[open_col])/df_data[price_ref_col]
    df_data['min_max_diff_ratio']  = (df_data[max_col] - df_data[min_col])/df_data[price_ref_col]

    # Data selection:
    df_data = df_data[df_data.index >= init_datetime]

    return df_data.dropna()

def get_macd_signal_hist(close_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # MACD:
    exp1 = close_series.ewm(span=12, adjust=False).mean()
    exp2 = close_series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    # Signal 
    signal = macd.ewm(span=9, adjust=False).mean()
    # Histogram
    histogram = macd - signal

    return macd, signal, histogram
