from datetime import datetime, timedelta
import joblib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from dtaidistance import dtw

import data.dataset as dtst

plt.rcParams.update({'font.size': 12})
np.random.seed(42)


class StockView:
    def __init__(self, data, modelpath='../models/model_enhanced_v3.joblib', price_ref='Close', ticker=''):
        self.data = data
        self.model = joblib.load(modelpath)
        self.price_ref = price_ref
        self.ticker = ticker

    def run_random_dates(self, n=10, wsize=20):
        chossen_dates = [datetime.strptime(x, '%Y-%m-%d') for x in np.random.choice(list(self.data['Date']), n)]
        for date in chossen_dates:
            self.run_daily_view(date, wsize)
    
    def run_daily_view(self, current_date: datetime, wsize=20, figsize=(15, 9)):
        count = 0
        while current_date not in self.data.index:
            current_date += timedelta(days = 1)
            count += 1
            if count >= 10:
                print("Chosen date is not available.")
                break
        current_date_str = str(current_date).split(' ')[0]
        init_time = current_date - timedelta(days = wsize)
        end_time = current_date + timedelta(days = wsize)
        df_tmp1 = self.data[(self.data.index >= init_time) & (self.data.index <= current_date)]
        df_tmp2 = self.data[(self.data.index >= current_date) & (self.data.index <= end_time)]

        plt.figure(figsize=figsize)

        plt.subplot(3,1,1)
        plt.title(f"{self.ticker} {current_date_str}")
        df_tmp1[self.price_ref].plot(label=f'{self.price_ref} price', zorder=2, marker='s', color='blue', alpha=0.75)
        df_tmp2[self.price_ref].plot(label=f'{self.price_ref} price (future values)', zorder=2, marker='s', color='gray', alpha=0.5)
        df_tmp1['sma9'].plot(label='sma 9 days', zorder=1, color='orange', alpha=0.5)
        df_tmp1['sma21'].plot(label='sma 21 days', zorder=1, color='green', alpha=0.5)
        df_tmp1['sma50'].plot(label='sma 50 days', zorder=1, color='purple', alpha=0.5)
        plt.xlim([init_time, end_time])
        # plt.ylim(marker_intervals)
        plt.xticks([])
        plt.ylabel(f'{self.price_ref} price [$]')
        plt.legend(loc='best', prop={'size': 10})
        plt.tight_layout()

        plt.subplot(3,1,2)
        df_tmp1['macd'].plot(label='MACD')
        df_tmp1['signal'].plot(label='Signal')
        plt.bar(df_tmp1.index, df_tmp1['histogram'], label='Histogram')
        plt.xlim([init_time, end_time])
        plt.xticks([])
        plt.ylabel(f'MACD/Signal/Histogram')
        plt.legend(loc='best', prop={'size': 10})
        plt.tight_layout()

        plt.subplot(3,1,3)
        df_tmp1['williams_r'].plot(label='williams_r')
        plt.plot([init_time, end_time], [-20, -20], linestyle='--', alpha=0.5)
        plt.plot([init_time, end_time], [-80, -80], linestyle='--', alpha=0.5)
        plt.xlim([init_time, end_time])
        plt.ylim([-100, 0])
        plt.ylabel(f'Williams %')
        plt.legend(loc='best', prop={'size': 10})
        plt.tight_layout()

        plt.show()

        print(f'{self.ticker} stock analysis at {current_date_str}')
        sma9gtsma21 = df_tmp1['sma9'] > df_tmp1['sma21']
        sma9gtsma50 = df_tmp1['sma9'] > df_tmp1['sma50']
        sma21gtsma50 = df_tmp1['sma21'] > df_tmp1['sma50']
        print("9 days S.M.A > 21 days S.M.A:", sma9gtsma21.values[-1])
        print("9 days S.M.A > 50 days S.M.A:", sma9gtsma50.values[-1])
        print("21 days S.M.A > 50 days S.M.A:", sma21gtsma50.values[-1])
        
        print("MACD:", round(df_tmp1['macd'].values[-1], 6))
        print("Signal:", round(df_tmp1['signal'].values[-1], 6))
        print("Histogram:", round(df_tmp1['histogram'].values[-1], 6))
        print("Williams %:", round(df_tmp1['williams_r'].values[-1], 6))

        df_tmp2['ticker'] = ''
        y_pred = self.model.predict(df_tmp2)
        if y_pred[0] == 0:
            trend_prediction = 'Down'
        elif y_pred[0] == 1:
            trend_prediction = 'Standing'
        else:
            trend_prediction = 'Up'
        print(f'Trend prediction: {trend_prediction}')


class StockSim:
    def __init__(self, ticker, chosen_date, days_interval=30, n_assets=3,
                 features_list=None, dist_metric='manhattan',
                 method='correlation', price_ref='Close', folderpath='../data/interim/stocks/'):
        self.ticker = ticker
        self.chosen_date = chosen_date
        self.days_interval = days_interval
        self.n_assets = n_assets
        self.start_date = chosen_date - timedelta(days=days_interval)
        self.method = method
        self.price_ref = price_ref
        self.folderpath = folderpath
        self.stock_data = dtst.load_stock_ticker(ticker, folderpath=folderpath)
        self.valid_stock_data = pd.DataFrame()
        
        if features_list:
            self.features_list = features_list
        else:
            self.features_list = ['macd', 'signal', 'histogram', 'williams_r',
                                  'pref_sma9_perc', 'pref_sma21_perc', 'pref_sma50_perc',
                                  'sma9gt21', 'sma9gt50', 'sma21gt50',
                                  'pref_ema9_perc', 'pref_ema21_perc', 'pref_ema50_perc',
                                  'ema9gt21', 'ema9gt50', 'ema21gt50',
                                  'open_close_diff_perc', 'open_close_diff_ratio', 'min_max_diff_ratio']
        self.dist_metric = dist_metric
        self.dist_matrix = None

        self.assets_files = os.listdir(folderpath)
        filename = f'{ticker}.csv'
        if filename in self.assets_files:
            self.assets_files.remove(filename)
        self.assets_tickers = [t.split('.')[0] for t in self.assets_files]

        self.assets_data = {}
        for asset_ticker in self.assets_tickers:
            self.assets_data[asset_ticker] = dtst.load_stock_ticker(asset_ticker, folderpath=folderpath)
        print(f'{len(self.assets_files)} other assets available.')
        
        self.valid_assets_data = {}
        self.selected_assets = []

    def run(self, method=None, days_interval=None):
        if method:
            self.method = method
        if days_interval:
            self.days_interval = days_interval
            self.start_date = self.chosen_date - timedelta(days=days_interval)
            
        # Selecting valid data:
        self.valid_stock_data = {}
        self.valid_stock_data = self.stock_data[(self.stock_data.index >= self.start_date) & (self.stock_data.index <= self.chosen_date)]
        for key,data in self.assets_data.items():
            data_tmp = data[(data.index >= self.start_date) & (data.index <= self.chosen_date)]
            if self.valid_stock_data.shape[0] == data_tmp.shape[0]:
                self.valid_assets_data[key] = data_tmp

        # Selecting the chosen method:
        if self.method in ['correlation' , 'c']:
            self._correlation_method()
        elif self.method in ['features_dist' , 'f']:
            self._features_dist()
        elif self.method in ['dtw' , 'w']:
            self._dtw_dist()
        else:
            self._correlation_method()

        # Plotting the results:
        self._plot_results()

    def _correlation_method(self):
        # Selecting price arrays:
        price_df = pd.DataFrame()
        price_df[self.ticker] = self.valid_stock_data[self.price_ref].tolist()
        for key,data in self.valid_assets_data.items():
            price_df[key] = data[self.price_ref].tolist()
        # Calculating the correlation matrix:
        corr_matrix = price_df.corr()
        corr_matrix = corr_matrix.sort_values(by=self.ticker, ascending=False)
        self.selected_assets = corr_matrix.index[1:self.n_assets+1].tolist()
    
    def _features_dist(self):
        '''
        The distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
        '''

        features_df = self.valid_stock_data.loc[[self.valid_stock_data.index[-1]], self.features_list]
        features_df = pd.concat([features_df] + [data.loc[[self.valid_stock_data.index[-1]], self.features_list] for data in self.valid_assets_data.values()])
        features_df = features_df.reset_index(drop=True)
        dist_matrix = pairwise_distances(features_df.values, metric=self.dist_metric)

        df_dist = pd.DataFrame()
        df_dist['assets'] = [self.ticker] + list(self.valid_assets_data.keys())
        df_dist['dist'] = dist_matrix[:, 0]
        df_dist = df_dist.sort_values(by='dist')
        self.selected_assets = df_dist['assets'].values[1:self.n_assets+1].tolist()
    
    def _dtw_dist(self):
        # Selecting price arrays:
        price_df = pd.DataFrame()
        price_df[self.ticker] = self.valid_stock_data[self.price_ref].tolist()/self.valid_stock_data[self.price_ref].max()
        for key,data in self.valid_assets_data.items():
            price_df[key] = data[self.price_ref].tolist()
        array_ref = price_df[self.ticker].values
        dtw_array = [dtw.distance(array_ref, price_df[col].values/price_df[col].values.max()) for col in price_df.columns]
        
        df_dtw = pd.DataFrame()
        df_dtw['assets'] = price_df.columns
        df_dtw['dist'] = dtw_array
        df_dtw = df_dtw.sort_values(by='dist')
        self.selected_assets = df_dtw['assets'].values[1:self.n_assets+1].tolist()
        self.df_dtw = df_dtw


    def _plot_results(self):
        plt.figure(figsize=(14, 5*len(self.selected_assets)))
        plt.subplot(len(self.selected_assets)+1, 1, 1)
        plt.title(f'Chosen ticker: {self.ticker}')
        plt.plot(self.valid_stock_data.index, self.valid_stock_data[self.price_ref], marker='s', alpha=0.75)
        for i,key in enumerate(self.selected_assets):
            plt.subplot(len(self.selected_assets)+1, 1, i+2)
            plt.title(f'Similar ticker: {key} (option {i+1})')
            plt.plot(self.valid_stock_data.index, self.valid_assets_data[key][self.price_ref], marker='s', alpha=0.75)
        plt.tight_layout()
        plt.show()
