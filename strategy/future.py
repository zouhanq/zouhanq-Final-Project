import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import ta

pro = ts.pro_api('dae2b8f31d48e585fb1d09b4f06ec3b83af43eb20cec39a2a2b71dce')

df = pro.df = pro.ft_mins(ts_code='IF.CFX', freq='30min', start_date='2022-04-25 09:30:00', end_date='2023-06-25 19:00:00')

# 全局变量
data_root = r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data'


def init_db():
    '''
    初始化股票数据库
    :return:
    '''
    # 1.获取所有股票代码
    futures = get_future_list()
    # 2.存储到csv文件中
    for code in futures:
        df = get_single_price(code, 'daily')
        export_data(df, code, 'price')
        print(code)
        print(df.head())


def get_future_list():
    """
    获取所有A股股票列表
    上海证券交易所.XSHG
    深圳证券交易所.XSHE
    :return: stock_list
    """
    stock_list = list(get_all_securities(['stock']).index)
    return stock_list




def export_data(data, filename, type, mode=None):
    """
    导出股票相关数据
    :param data:
    :param filename:
    :param type: 股票数据类型，可以是：price、finance
    :param mode: a代表追加，none代表默认w写入
    :return:
    """
    file_root = data_root + type + '/' + filename + '.csv'
    data.index.names = ['date']
    if mode == 'a':
        data.to_csv(file_root, mode=mode, header=False)
        # 删除重复值
        data = pd.read_csv(file_root)  # 读取数据
        data = data.drop_duplicates(subset=['date'])  # 以日期列为准
        data.to_csv(file_root, index=False)  # 重新写入
    else:
        data.to_csv(file_root)  # 判断一下file是否存在 > 存在：追加 / 不存在：保持

    print('已成功存储至：', file_root)




def transfer_price_freq(data, time_freq):
    """
    将数据转换为制定周期：开盘价（周期第1天）、收盘价（周期最后1天）、最高价（周期内）、最低价（周期内）
    :param data:
    :param time_freq:
    :return:
    """
    df_trans = pd.DataFrame()
    df_trans['open'] = data['open'].resample(time_freq).first()
    df_trans['close'] = data['close'].resample(time_freq).last()
    df_trans['high'] = data['high'].resample(time_freq).max()
    df_trans['low'] = data['low'].resample(time_freq).min()

    return df_trans

def calculate_change_pct(data):
    """
    涨跌幅 = (当期收盘价-前期收盘价) / 前期收盘价
    :param data: dataframe，带有收盘价
    :return: dataframe，带有涨跌幅
    """
    data['close_pct'] = (data['close'] - data['close'].shift(1)) \
                        / data['close'].shift(1)
    return data


def get_tushare_data(ts_code, freq, start_date, end_date, file_path):
    """
    Fetches data from Tushare and stores it into a CSV file.

    :param ts_code: str, the Tushare code of the asset
    :param freq: str, frequency of the data (e.g., '1min', '5min', '15min', '30min', '60min', 'D', 'W', 'M')
    :param start_date: str, start date in 'YYYY-MM-DD HH:MM:SS' format
    :param end_date: str, end date in 'YYYY-MM-DD HH:MM:SS' format
    :param api_key: str, your Tushare API key
    :param file_path: str, the path to the CSV file to save the data
    :return: DataFrame, the fetched data
    """
    # Initialize Tushare
    pro = ts.pro_api('dae2b8f31d48e585fb1d09b4f06ec3b83af43eb20cec39a2a2b71dce')

    # Fetch the data
    df = pro.ft_mins(ts_code=ts_code, freq=freq, start_date=start_date, end_date=end_date)

    # Ensure the necessary columns exist
    required_columns = ['trade_time', 'open', 'high', 'low', 'close', 'vol']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert 'trade_time' column to datetime if necessary
    df['trade_time'] = pd.to_datetime(df['trade_time'])

    # Save to CSV file
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

    return df

if __name__ == "__main__":
    # Load your CSV data
    get_tushare_data('IF.CFX','1min', '2022-04-25 09:30:00', '2023-06-25 19:00:00', r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\sif.csv')