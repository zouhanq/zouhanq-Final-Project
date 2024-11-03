import base as strat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import ta
import ma_strategy as ma
import time
from openctp_ctp import tdapi, mdapi
import signal
import sys




class CMdImpl(mdapi.CThostFtdcMdSpi):
    def __init__(self, md_front,historical_data, td):
        super().__init__()
        self.md_front = md_front
        self.api = None
        self.csv_file_path = r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\mink.csv'
        self.data =  historical_data # Initialize with historical data from CSV
        self.td = td  # Reference to the trading API class

    def Run(self):
        self.api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi()
        self.api.RegisterFront(self.md_front)
        self.api.RegisterSpi(self)
        self.api.Init()

    def OnFrontConnected(self):
        print("Market Data Front Connected")
        req = mdapi.CThostFtdcReqUserLoginField()
        self.api.ReqUserLogin(req, 0)

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"Login failed: {pRspInfo.ErrorMsg}")
            return
        print(f"Login succeed. Trading Day: {pRspUserLogin.TradingDay}")
        self.api.SubscribeMarketData(["IF2408".encode('utf-8')], 1)

    def OnRtnDepthMarketData(self, pDepthMarketData):
        print(f"Received market data: {pDepthMarketData.InstrumentID} - {pDepthMarketData.LastPrice} - {pDepthMarketData.Volume}")
        # Append received data to the DataFrame
        new_data = pd.DataFrame({
            'Time': [pd.to_datetime(f"{pDepthMarketData.TradingDay} {pDepthMarketData.UpdateTime}.{pDepthMarketData.UpdateMillisec}")],
            'InstrumentID': [pDepthMarketData.InstrumentID],
            'volume': [pDepthMarketData.Volume],
            'open': [pDepthMarketData.OpenPrice],
            'high': [pDepthMarketData.HighestPrice],
            'low': [pDepthMarketData.LowestPrice],
            'close': [pDepthMarketData.LastPrice]  
        }) 
        self.data = pd.concat([self.data, new_data]).reset_index(drop=True)

        # Save the updated data to CSV
    
        self.process_market_data()
    
    def process_market_data(self):
        data = self.data.copy()
        data = ma.ma_strategy(data)  # Apply the moving average strategy

        # Place orders based on signals
        for index, row in data.iterrows():
            if row['buy_signal'] == 1:
                self.td.place_order("SHFE", 'IF2408', tdapi.THOST_FTDC_D_Buy, tdapi.THOST_FTDC_OF_Open, row['close'], 1)
            elif row['sell_signal'] == -1:
                self.td.place_order("SHFE", 'IF2408', tdapi.THOST_FTDC_D_Sell, tdapi.THOST_FTDC_OF_Close, row['close'], 1)
    def save_data(self):
        """Save the market data to a CSV file."""
        self.data.to_csv(r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\mink.csv', index=False)



class TdImpl(tdapi.CThostFtdcTraderSpi):
    def __init__(self, host, broker, user, password, appid, authcode):
        super().__init__()
        self.broker = broker
        self.user = user
        self.password = password
        self.appid = appid
        self.authcode = authcode
        self.OrderRef = 0
        
        self.api: tdapi.CThostFtdcTraderApi = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi()
        self.api.RegisterSpi(self)
        self.api.RegisterFront(host)
        self.api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
        self.api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)

    def Run(self):
        self.api.Init()

    def OnFrontConnected(self):
        print("Trading Front Connected")
        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.AppID = self.appid
        req.AuthCode = self.authcode
        self.api.ReqAuthenticate(req, 0)

    def OnRspAuthenticate(self, pRspAuthenticateField, pRspInfo, nRequestID, bIsLast):
        if pRspInfo and pRspInfo.ErrorID != 0:
            print(f"Authentication failed: {pRspInfo.ErrorMsg}")
            exit(-1)
        print("Authenticated successfully")
        req = tdapi.CThostFtdcReqUserLoginField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.Password = self.password
        self.api.ReqUserLogin(req, 0)

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        if pRspInfo and pRspInfo.ErrorID != 0:
            print(f"Login failed: {pRspInfo.ErrorMsg}")
            exit(-1)
        print("Login successful")
        # Ready to place orders

    def place_order(self, exchange_id, instrument_id, direction, offset, price, volume):
        req = tdapi.CThostFtdcInputOrderField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.ExchangeID = exchange_id
        req.InstrumentID = instrument_id
        req.Direction = direction
        req.CombOffsetFlag = offset
        req.OrderPriceType = tdapi.THOST_FTDC_OPT_LimitPrice
        req.LimitPrice = price
        req.VolumeTotalOriginal = volume
        req.TimeCondition = tdapi.THOST_FTDC_TC_GFD
        req.VolumeCondition = tdapi.THOST_FTDC_VC_AV
        req.MinVolume = 1
        req.OrderRef = str(self.OrderRef)
        self.OrderRef += 1
        req.ContingentCondition = tdapi.THOST_FTDC_CC_Immediately
        req.ForceCloseReason = tdapi.THOST_FTDC_FCC_NotForceClose
        self.api.ReqOrderInsert(req, 0)

    def cancel_order(self, exchange_id, instrument_id, order_sys_id, front_id, session_id, order_ref):
        req = tdapi.CThostFtdcInputOrderActionField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.ExchangeID = exchange_id
        req.InstrumentID = instrument_id
        req.OrderSysID = order_sys_id
        req.FrontID = front_id
        req.SessionID = session_id
        req.OrderRef = order_ref
        req.ActionFlag = tdapi.THOST_FTDC_AF_Delete
        self.api.ReqOrderAction(req, 0)


def read_historical_data():
    """Read historical market data from a CSV file."""
    df = pd.read_csv(r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\mink.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def signal_handler(md):
    """Handle termination signals to save data before exiting."""
    print("Program is terminating. Saving data...")
    md.save_data()
    sys.exit(0)


if __name__ == "__main__":
    csv_file_path = r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\mink.csv'
    historical_data = read_historical_data()
    td = TdImpl("tcp://180.168.146.187:10130", "9999", "228402", "woshiqi001216@", "simnow_client_test", "0000000000000000")
    td.Run()

    md = CMdImpl("tcp://180.168.146.187:10211", historical_data, td)
    md.csv_file_path = csv_file_path
    md.Run()

    # Setup signal handling to save data on termination
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(md))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(md))

    # Keep the main thread alive
    while True:
        time.sleep(1)