import pandas_datareader.data as web
import pandas as pd
import numpy as np
from talib import RSI, BBANDS
import matplotlib.pyplot as plt

# Tried Technical Anlaysis strategy mentioned by Kyle Li in the post.
https://towardsdatascience.com/trading-strategy-technical-analysis-with-python-ta-lib-3ce9d6ce5614
# The code samples are joined and make it working version :

start = '2015-04-22'
end = '2017-04-22'

symbol = 'MCD'
max_holding = 100
price = web.DataReader(name=symbol, data_source='quandl', start=start, end=end)
price = price.iloc[::-1]
price = price.dropna()
close = price['AdjClose'].values
rsi = RSI(close, timeperiod=14)
price['RSI'] = rsi

up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
price['bb_up'] = up;
price['bb_mid'] = mid;
price['bb_low'] = low;

price = price.dropna()

isHoldingFull = False
holdings = pd.DataFrame(index=price.index, data={'Holdings': np.array([np.nan] * price.index.shape[0])})
for index, row in price.iterrows():
  adjClose = row[10]
  r_s_i = row[12]
  b_up = row[13]
  b_low = row[15]
  bbp = (adjClose - b_low) / (b_up - b_low)
  if((not isHoldingFull) & (r_s_i < 30) & (bbp < 0)):
    holdings.loc[index, 'Holdings'] = 100
    isHoldingFull = True;
    #if(index is '2015-08-24 00:00:00'):
  elif((r_s_i > 70) & (bbp > 1)):
    holdings.loc[index, 'Holdings'] = 0
    isHoldingFull = False;
  elif(isHoldingFull):
    holdings.loc[index, 'Holdings'] = 100
  else:
    holdings.loc[index, 'Holdings'] = 0


holdings.ffill(inplace=True)
holdings.fillna(0, inplace=True)
holdings['Order'] = holdings.diff()
holdings.dropna(inplace=True)


fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax0.plot(price.index.values, price['AdjClose'], label='AdjClose')
ax0.set_xlabel('Date')
ax0.set_ylabel('AdjClose')
ax0.grid()
for day, holding in holdings.iterrows():
    order = holding['Order']
    if order > 0:
        ax0.scatter(x=day, y=price.loc[day, 'AdjClose'], color='green')
    elif order < 0:
        ax0.scatter(x=day, y=price.loc[day, 'AdjClose'], color='red')

ax1.plot(price.index.values, price['RSI'], label='RSI')
ax1.fill_between(price.index.values, y1=30, y2=70, color='#adccff', alpha='0.3')
ax1.set_xlabel('Date')
ax1.set_ylabel('RSI')
ax1.grid()

ax2.plot(price.index.values, price['bb_up'], label='BB_up')
ax2.plot(price.index.values, price['AdjClose'], label='AdjClose')
ax2.plot(price.index.values, price['bb_low'], label='BB_low')
ax2.fill_between(index, y1=price['bb_low'], y2=price['bb_up'], color='#adccff', alpha='0.3')
ax2.set_xlabel('Date')
ax2.set_ylabel('Bollinger Bands')
ax2.grid()

fig.tight_layout()
plt.show()
