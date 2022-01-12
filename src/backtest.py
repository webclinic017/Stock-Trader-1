import vectorbt as vbt
import pandas as pd
import os

if __name__ == "__main__":
#read all files in stocks dir
  for file in os.listdir("stocks"):
      data = pd.read_csv(f"stocks/{file}")

      # convert close to a series object
      series = pd.Series(data["Close"])
      
      # set date as index
      series.index = data["Date"]

      # create a backtest object
      fast_ma = vbt.MA.run(series, 21)
      slow_ma = vbt.MA.run(series, 5)
      entries = fast_ma.ma_above(slow_ma, crossover=True)
      exits = fast_ma.ma_below(slow_ma, crossover=True)
      pf = vbt.Portfolio.from_signals(series, entries, exits, init_cash=10000)

      # create a backtest for buy and hold
      pf_hold = vbt.Portfolio.from_holding(series, init_cash=10000)

      print(file.split(".")[0])
      print(pf.total_profit())
      print(pf.total_return())
      print(f"Buy and hold profit: {pf_hold.total_profit()}")