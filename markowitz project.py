import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

num_trading_days=252
num_portfolios =10000

stocks=['APPL','WMT','TSLA','GE','AMZN','DB']

start_date= '2012-01-01'
end_date= '2017-01-01'

def download_data():
    stock_data= {}
    for stock in stocks:
        ticker=yf.Ticker(stock)
        stock_data[stock]=ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)
def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_return(data):
    log_return=np.log(data/data.shift(1))
    return log_return[1:] #to remove NA value

def show_statistics(returns):
    print(returns.mean()*num_trading_days)
    print(returns.cov() * num_trading_days)

def show_mean_variance(returns,weights):
    #we are after the annual return
    portfolio_return=np.sum(returns.mean()*weights)*num_trading_days
    portfolio_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*num_trading_days,weights)))
    print(portfolio_return)
    print(portfolio_volatility)

def show_portfolios(returns,volatilities):
    plt.figure(figsize=(10,6))
    plt.scatter(volatilities,returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected return')
    plt.colorbar(label='sharpe ratio')
    plt.show()

def generate_portfolios(returns):
    portfolio_means =[]
    portfolio_risks =[]
    portfolio_weights =[]

    for _ in range(num_portfolios):
        w= np.random.random(len(stocks))
        w/=np.sum(w) #ensure sum equals to 1
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean()*w)*num_trading_days)
        portfolio_risks.append(np.sqrt(np.dot(w.T,np.dot(returns.cov()*num_trading_days,w))))
    return np.array(portfolio_weights),np.array(portfolio_means),np.array(portfolio_risks)

#if __name__ == '__main__':
#    dataset=download_data()
#    show_data(dataset)

if __name__ == '__main__':
    dataset=(download_data())
    show_data(dataset)
    log_daily_returns= calculate_return(dataset)
    show_statistics(log_daily_returns)

    weights, means,risks=generate_portfolios(log_daily_returns)
    show_portfolios(means,risks)


