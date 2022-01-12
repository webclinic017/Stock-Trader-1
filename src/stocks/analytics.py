import joblib
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np


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
