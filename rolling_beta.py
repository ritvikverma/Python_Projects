import pandas as pd
from data_access import *
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np
import mpl_finance as mplf


class BetaCalculation:  # first array is date_time, second array is stock, third array is index price, data in price
    def __init__(self, stock_code, from_date=None, to_date=None, exchange='sehk'):
        self.stock_code = stock_code
        self.exchange = exchange
        self.from_date = from_date
        self.to_date = to_date
        self.stock_data = get_stock_price(stock_code=self.stock_code,
                                          exchange=self.exchange,
                                          date_from=self.from_date,
                                          date_to=self.to_date).loc[:, ['date_time', 'open', 'high', 'low', 'close']]#only select those specific values
        if exchange == 'sehk': #checks the exact market
            self.stock_index = 'HSI'
        elif exchange == 'nasdaq':
            self.stock_index = 'IXIC'
        elif exchange == 'nyse':
            self.stock_index = 'GSPC'
        self.index_data = self.get_index_data()
        self.stock_data = self.stock_data.loc[self.stock_data.date_time.isin(self.index_data.date_time.tolist()), :]
        self.stock_data = self.stock_data.reset_index(drop=True)

    def get_stock_beta(self, stock_ret_series, index_ret_series): #get the beta for the stock
        temp_df = pd.DataFrame({self.stock_code: stock_ret_series, self.stock_index: index_ret_series})
        cov_df = temp_df.cov()
        return_beta = cov_df.iloc[1, 0] / cov_df.iloc[1, 1]
        return return_beta

    def get_index_data(self): #gets all the data for index, in specified columns
        index_data = get_stock_price(stock_code=self.stock_index, #First get stock price
                                     exchange='index',
                                     date_from=self.from_date,
                                     date_to=self.to_date).loc[:, ['date_time', 'open', 'high', 'low', 'close']]
        return index_data

    def roll_the_beta(self, freq=22, column_title=None, return_type='df'):
        return_df = self.stock_data.copy()
        if column_title is None:
            column_title = str(freq) + 'D_beta'
        return_df[column_title] = None
        date_time_list = return_df.loc[:, 'date_time'].tolist()
        n = date_time_list.__len__()
        index_df = self.index_data.copy()
        stock_df = self.stock_data.copy()
        index_df.loc[:, 'ret_ser'] = self.index_data.close.pct_change()
        stock_df.loc[:, 'ret_ser'] = self.stock_data.close.pct_change()
        for i in range(n):
            if i < freq-1:
                continue
            pass_index_ret = index_df.loc[
                (index_df.date_time >= date_time_list[i - freq + 1]) &
                (index_df.date_time <= date_time_list[i]), 'ret_ser']
            pass_stock_ret = stock_df.loc[
                (stock_df.date_time >= date_time_list[i - freq + 1]) &
                (stock_df.date_time <= date_time_list[i]), 'ret_ser']
            period_beta = self.get_stock_beta(pass_stock_ret, pass_index_ret)
            return_df.loc[return_df.date_time == date_time_list[i], column_title] = period_beta
        if return_type == 'df':
            return return_df
        elif return_type == 'series':
            return return_df.loc[:, column_title]
        elif return_type == 'list':
            return return_df.loc[:, column_title].tolist()


def plot_beta(price_df, save_fig=False): #plot the beta
    columns_list = price_df.columns.tolist()
    beta_columns_list = []
    for i in range(columns_list.__len__()):
        if columns_list[i].find('beta') != -1:
            beta_columns_list.append(columns_list[i])
    beta_df = price_df.loc[:, beta_columns_list]
    hori_size = np.log10(round(len(price_df)))
    fig, ax1 = plt.subplots(figsize=(hori_size * 20, 20))
    color = 'tab:blue'
    ax1.set_xlabel('date_time')
    ax1.set_ylabel('stock_price', color=color)
    ax1.plot(price_df.date_time, price_df.close, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()

    x_date = price_df.date_time.tolist()
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax1.xaxis.set_major_formatter(ticker.IndexFormatter(x_date))
    fig.autofmt_xdate()

    beta_color = 'tab:red'
    ax2.set_ylabel('rolling_beta')
    ax2.plot(price_df.date_time, beta_df, color=beta_color)
    ax2.tick_params(axis='y', labelcolor=beta_color)

    fig.tight_layout()
    plt.show()
    if save_fig:
        plt.savefig('target_graph.png')
    return None


if __name__ == '__main__': #Sees if this is being run directly, otherwise doesn't run
    target_stock_code = 700 #Target stock code
    index_code = 'HSI' #Type of stock
    start_date = '2015-09-02'#Start date
    end_date = '2018-09-09' #End date
    try_bc = BetaCalculation(700, start_date, end_date) #Current beta calculation object
    # beta_list = ['1W_beta', '1M_beta', '3M_beta', '6M_beta', '1Y_beta', '3Y_beta'] #Stock title list
    # beta_days_list = [5, 21, 63, 126, 252, 756] #Actual days for consideration, but not used
    summary_df = try_bc.stock_data.copy() #Summary of beta values across timeframes, copy
    index_df = try_bc.index_data.copy() #Summary of index values across timeframes, copy
    summary_df = summary_df.loc[summary_df.date_time.isin(index_df.date_time.tolist()), :]
    summary_df.loc[:, '1W_beta'] = try_bc.roll_the_beta(freq=5, return_type='list') #Add 5 day beta roll across timeframe to summary DF
    summary_df.loc[:, '1M_beta'] = try_bc.roll_the_beta(freq=21, return_type='list')#Add 21 day beta roll across timeframe to summary DF
    summary_df.loc[:, '3M_beta'] = try_bc.roll_the_beta(freq=63, return_type='list')#Add 63 day beta roll across timeframe to summary DF
    summary_df.loc[:, '6M_beta'] = try_bc.roll_the_beta(freq=126, return_type='list')#Add 126 day beta roll across timeframe to summary DF
    summary_df.loc[:, '1Y_beta'] = try_bc.roll_the_beta(freq=252, return_type='list')#Add 252 day beta roll across timeframe to summary DF
    summary_df.loc[:, '3Y_beta'] = try_bc.roll_the_beta(freq=756, return_type='list')#Add 756 day beta roll across timeframe to summary DF
    print(summary_df) #prints the summary of the beta values across timeframes
    # print(try_bc.index_data) #prints all the index data
    # plot_beta(summary_df) #plots the summary of the dataframe
    #print(try_bc.index_data)

