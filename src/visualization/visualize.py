import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_assets_timeline(assets_dict, subplotsize=(10, 6)):
    if len(assets_dict) > 10:
        assets_dict_ = {}
        for key,val in assets_dict.items():
            assets_dict_[key] = val
            if len(assets_dict_) >= 10:
                break
        assets_dict = assets_dict_

    plt.figure(figsize=(subplotsize[0], subplotsize[1]*len(assets_dict)))
    for i, (asset_name, df) in enumerate(assets_dict.items()):
        plt.subplot(len(assets_dict), 1, i+1)
        plt.plot(df.index, df['Close'].values, label=asset_name, alpha=0.75)
        if len(assets_dict) <= 10:
            plt.legend(loc='best')
        plt.ylabel('Asset value [$]')
    plt.tight_layout()
    plt.show()


def plot_candlesticks(stock_data, title='', yaxis_title=''):
    candlestick = go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'], high=stock_data['High'],
                                 low=stock_data['Low'], close=stock_data['Close'],
                                )
    fig = go.Figure(data=[candlestick])
    # fig.update_layout()
    fig.update_layout(title=title,
                      width=1000, height=600,
                    #   xaxis_rangeslider_visible=False,
                      yaxis_title=yaxis_title,
    )
    fig.show()


def plot_macd(stock_data, price_ref_col):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    stock_data[price_ref_col].plot(label=price_ref_col, zorder=2)
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.ylabel('stock price [$]')
    plt.grid()
    plt.subplot(2, 1, 2)
    stock_data['macd'].plot(label='MACD')
    stock_data['signal'].plot(label='Signal')
    plt.bar(stock_data.index, stock_data['histogram'], label='Histogram')
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_williams_r(stock_data, price_ref_col):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    stock_data[price_ref_col].plot(label=price_ref_col, zorder=2)
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.ylabel('stock price [$]')
    plt.grid()
    plt.subplot(2, 1, 2)
    stock_data['williams_r'].plot(label='williams_r')
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_sma_features(stock_data, price_ref_col):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    stock_data[price_ref_col].plot(label=price_ref_col, zorder=2)
    stock_data['sma9'].plot(label='sma9', zorder=2)
    stock_data['sma21'].plot(label='sma21', zorder=2)
    stock_data['sma50'].plot(label='sma50', zorder=2)
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.ylabel('stock price [$]')
    plt.grid()
    plt.subplot(2, 1, 2)
    stock_data['pref_sma9_perc'].plot(label='pref_sma9_perc')
    stock_data['pref_sma21_perc'].plot(label='pref_sma21_perc')
    stock_data['pref_sma50_perc'].plot(label='pref_sma50_perc')
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_ema_features(stock_data, price_ref_col):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    stock_data[price_ref_col].plot(label=price_ref_col, zorder=2)
    stock_data['ema9'].plot(label='ema9', zorder=2)
    stock_data['ema21'].plot(label='ema21', zorder=2)
    stock_data['ema50'].plot(label='ema50', zorder=2)
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.ylabel('stock price [$]')
    plt.grid()
    plt.subplot(2, 1, 2)
    stock_data['pref_ema9_perc'].plot(label='pref_ema9_perc')
    stock_data['pref_ema21_perc'].plot(label='pref_ema21_perc')
    stock_data['pref_ema50_perc'].plot(label='pref_ema50_perc')
    plt.xlim([stock_data.index.min(), stock_data.index.max()])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



def plot_features_hist(df, features_list, split_col='y_target', figsize=(14, 4)):
    for col in features_list:
        plt.figure(figsize=figsize)
        plt.title(col)
        for val in df[split_col].unique():
            df_tmp = df[df['y_target'] == val]
            plt.hist(df_tmp[col], bins=30, alpha=0.5)
        plt.show()
