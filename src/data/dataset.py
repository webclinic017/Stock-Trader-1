import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rawdata(folderpath, n_files=None, selected_files=None):
    # Getting the files pathlist:
    if selected_files is None:
        filenames_list = os.listdir(folderpath)
        if n_files is None:
            n_files = len(filenames_list)
        filenames_list = filenames_list[0:n_files]
    else:
        filenames_list = selected_files
    filepaths_list = [os.path.join(folderpath, filename) for filename in filenames_list]

    # Loading all the files in the files_pathlist
    assets_dict = {}
    for filename, filepath in zip(filenames_list, filepaths_list):
        df = pd.read_csv(filepath)
        df.index = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')).values
        assets_dict[filename.split('.')[0]] = df
    return assets_dict


def trend_labeling(df_stock, price_ref_col, title='', figsize=(12, 6), plot_results=False, diff_thold=1e-3):
    """
    target values:
    0: down trend
    1: standing
    2: up trend
    """
    target_labels = ['down trend', 'standing', 'up trend']

    df_stock = df_stock.copy()
    y_target = np.ones(df_stock.shape[0])
    # Calculating the moving average:
    y_array = df_stock[price_ref_col].values
    y_array_ma = np.zeros(df_stock.shape[0])
    for i,y_val in enumerate(y_array):
        if (i-2) >= 0:
            try:
                y_array_ma[i] = sum([y_array[i-2], y_array[i-1], y_val, y_array[i+1], y_array[i+2]])/5
            except IndexError:
                y_array_ma[i] = y_val
        else:
            y_array_ma[i] = y_val
    df_stock[f'{price_ref_col}_ma'] = y_array_ma
    
    # Setting the target trend value:
    for i,y_val_ma in enumerate(y_array_ma):
        if (i-1) >= 0:
            try:
                diff1 = abs(y_array_ma[i-1] - y_val_ma)/y_array_ma[i-1]
                diff2 = abs(y_val_ma - y_array_ma[i+1])/y_val_ma
                if y_array_ma[i-1] < y_val_ma < y_array_ma[i+1] and diff1 > diff_thold and diff2 > diff_thold:
                    y_target[i] = 2
                elif y_array_ma[i-1] > y_val_ma > y_array_ma[i+1] and diff1 > diff_thold and diff2 > diff_thold:
                    y_target[i] = 0
            except IndexError:
                pass
    df_stock['y_trend'] = y_target

    if plot_results:
        print(f'{title} df shape:', df_stock.shape)
        print(df_stock['y_trend'].value_counts())

        df_stock_ = df_stock[0:250]
        plt.figure(figsize=figsize)
        plt.subplot(2,1,1)
        plt.plot(df_stock_[price_ref_col], label=title)
        plt.plot(df_stock_[f'{price_ref_col}_ma'], color='gray', label=f'{title}_ma')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(df_stock_[f'{price_ref_col}_ma'], color='gray', label=f'{title}_ma')
        y_trend_unique = list(df_stock_['y_trend'].unique())
        y_trend_unique.sort()
        for t in y_trend_unique:
            df_stock_tmp = df_stock_[df_stock_['y_trend'] == t]
            plt.scatter(df_stock_tmp.index, df_stock_tmp[f'{price_ref_col}_ma'], label=target_labels[int(t)], alpha=0.75, s=40)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return y_target


def load_stocks_data(folderpath):
    train_path = f'{folderpath}/train/'
    df_train = pd.concat([df for df in load_rawdata(train_path).values()])
    test_path = f'{folderpath}/test/'
    df_test = pd.concat([df for df in load_rawdata(test_path).values()])

    # Eliminating inf and nan values:
    df_train = df_train.replace(np.inf, np.nan).dropna()
    df_test = df_test.replace(np.inf, np.nan).dropna()
    # Replacing boolean features to int values
    df_train['sma9gt21'] = df_train['sma9gt21'].astype(int)
    df_train['sma9gt50'] = df_train['sma9gt50'].astype(int)
    df_train['sma21gt50'] = df_train['sma21gt50'].astype(int)
    df_train['ema9gt21'] = df_train['ema9gt21'].astype(int)
    df_train['ema9gt50'] = df_train['ema9gt50'].astype(int)
    df_train['ema21gt50'] = df_train['ema21gt50'].astype(int)
    df_test['sma9gt21'] = df_test['sma9gt21'].astype(int)
    df_test['sma9gt50'] = df_test['sma9gt50'].astype(int)
    df_test['sma21gt50'] = df_test['sma21gt50'].astype(int)
    df_test['ema9gt21'] = df_test['ema9gt21'].astype(int)
    df_test['ema9gt50'] = df_test['ema9gt50'].astype(int)
    df_test['ema21gt50'] = df_test['ema21gt50'].astype(int)

    return df_train, df_test


def load_stock_ticker(ticker: str):
    filepath = f'../data/interim/stocks/{ticker}.csv'
    df = pd.read_csv(filepath)
    df.index = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')).values
    df['sma9gt21'] = df['sma9gt21'].astype(int)
    df['sma9gt50'] = df['sma9gt50'].astype(int)
    df['sma21gt50'] = df['sma21gt50'].astype(int)
    df['ema9gt21'] = df['ema9gt21'].astype(int)
    df['ema9gt50'] = df['ema9gt50'].astype(int)
    df['ema21gt50'] = df['ema21gt50'].astype(int)

    return df
