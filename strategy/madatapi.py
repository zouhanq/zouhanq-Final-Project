from openctp_ctp import tdapi, mdapi


class CMdImpl(mdapi.CThostFtdcMdSpi):
    def __init__(self, md_front):
        # Initialize the market data SPI and set the front address
        mdapi.CThostFtdcMdSpi.__init__(self)
        self.md_front = md_front
        self.api = None

    def Run(self):
        # Create the market data API instance and register the front and SPI
        self.api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi()
        self.api.RegisterFront(self.md_front)
        self.api.RegisterSpi(self)
        self.api.Init()

    def OnFrontConnected(self):
        # Callback for when the connection to the server is established
        print("OnFrontConnected")
        req = mdapi.CThostFtdcReqUserLoginField()
        # Send a login request (no need for user ID and password for market data)
        self.api.ReqUserLogin(req, 0)

    def OnFrontDisconnected(self, nReason: int):
        # Callback for when the connection to the server is disconnected
        print(f"OnFrontDisconnected.[nReason={nReason}]")

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        # Callback for the user login response
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"Login failed. {pRspInfo.ErrorMsg}")
            return
        print(f"Login succeed.{pRspUserLogin.TradingDay}")
        # Subscribe to market data for a specific instrument
        self.api.SubscribeMarketData(["au2406".encode('utf-8')], 1)

    def OnRtnDepthMarketData(self, pDepthMarketData):
        # Callback for receiving market data updates
        print(f"{pDepthMarketData.InstrumentID} - {pDepthMarketData.LastPrice} - {pDepthMarketData.Volume}")

if __name__ == '__main__':
    # Create an instance of the market data implementation and run it
    md = CMdImpl("tcp://180.168.146.187:10131")
    md.Run()
    input("Press enter key to exit.")