import random
import threading
import time
import datetime as dt

from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import requests
import arrow
import datetime
import scipy.stats as si
from math import *
from django.conf import settings



import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request



clientId = 0

class tradingThread(threading.Thread):
    def __init__(self, util, inputArg):
        threading.Thread.__init__(self)
        self.inputArg = inputArg
        self.util = util

    def run(self):
        self.util.getTrade(self.inputArg)

class Util:
    timeframe = "1 min"
    symbol_1 = 'AAPL'
    expiration_date_call = '20200110'
    strike_price_call = '310'

    expiration_date_put = '20200110'
    strike_price_put = '300'

    # set parameters for risk free rate, and time expiration, for Black Scholes Pricing Model Function:
    rf_rate = 0.017
    # risk free rate, normally whatever 10 year treasury yield is

    time_exp = 0.019
    # (30.42 / 365.0) is roughly one month, (7/365) is roughly 1 week, 3/365 is roughly 3 days

    expiration_premium_symbol_1_call = 1.023
    num_contracts = 1
    # specify number of contracts you want to trade
    # set percent thresholds for long and short side needed for ai accuracy on historical data for trade to trigger:
    ai_pct_threshold_long = 0.55
    ai_pct_threshold_short = 0.50
    ib = None

    def onError(self, msg):
        print("my msg -------", msg)

    def __init__(self):
        self.bRunning = False
        self.ib = IB()
        self.id = random.random()

        # self.ib.connect(settings.TRADER_HOST, settings.TRADER_PORT, clientId=settings.TRADER_CLIENTID)
        return

    def getSheet(self):
        sheet = {"expiration_call": '20200207',
                 "strike_call": '325',
                 "expiration_put": '20200207',
                 "strike_put": '312.50',
                 "ten_years_yield": '1.84',
                 "time_exp": '0.049',
                 "opt_diff": '1.020',
                 "num_of_contracts": '0',
                 "long_threshold": '0.58',
                 "short_threshold": '0.50'}
        return sheet

    def getTrade(self, inputArg):
        self.bRunning = True
        try :
            # Connect to API
            if self.ib.isConnected():
                pass
            else:
                globals()['clientId'] = (globals()['clientId'] + 1) % 32
                self.ib.connect(settings.TRADER_HOST, settings.TRADER_PORT, clientId=globals()['clientId'])
                # self.ib.connect('127.0.0.1', 7496, clientId=globals()['clientId'])

    ###### USER PARAMETERS ######
            self.timeframe = "1 min"
            # Global Variablespip
            self.pricedata_1 = None
    ########################################################################################################################
    # ##############################################################################
    ########################################################################################################################
    # ##############################################################################
    ########################################################################################################################
    # ##############################################################################
    ########################################################################################################################
    # ##############################################################################
        # INPUTS FOR ALGORITHM:

            self.symbol_1 = 'AAPL' if inputArg["symbol_1"] is None else inputArg["symbol_1"]
            self.expiration_date_call = '20200110' if inputArg["expiration_call"] is None else inputArg["expiration_call"]
            self.strike_price_call = '310' if inputArg["strike_call"] is None else inputArg["strike_call"]

            self.expiration_date_put = '20200110' if inputArg["expiration_put"] is None else inputArg["expiration_put"]

            self.strike_price_put = '300' if inputArg["strike_put"] is None else inputArg["strike_put"]

            # set parameters for risk free rate, and time expiration, for Black Scholes Pricing Model Function:
            self.rf_rate = 0.017 if inputArg["ten_years_yield"] is None else float(inputArg["ten_years_yield"])
            # risk free rate, normally whatever 10 year treasury yield is

            self.time_exp = 0.019 if inputArg["time_exp"] is None else float(inputArg["time_exp"])
            # (30.42 / 365.0) is roughly one month, (7/365) is roughly 1 week, 3/365 is roughly 3 days

            self.expiration_premium_symbol_1_call = 1.023 if inputArg["opt_diff"] is None else float(inputArg["opt_diff"])
            # means strike price for the call is approx. 2.3% above current market price ,
            # adjust accordingly to whatever fits your trading parameters profile. So if AAPL was trading
            # at approx 303 at time of these inputs, and our strike_price for call was set to 310, that would be approx.
            # 2.3% above the current market price. Which is why it is set now to 1.023 (2.3% increase
            # from current market price levels when calculated in the BSM model function below,
            # which is why there is a 1 before the .023)

            self.num_contracts = 1 if inputArg["num_of_contracts"] is None else float(inputArg["num_of_contracts"])
            # specify number of contracts you want to trade

            # set percent thresholds for long and short side needed for ai accuracy on historical data for trade to trigger:
            self.ai_pct_threshold_long = 0.55 if inputArg["long_threshold"] is None else float(inputArg["long_threshold"])
            self.ai_pct_threshold_short = 0.50 if inputArg["short_threshold"] is None else float(inputArg["short_threshold"])

    #######################################################################################################################
    ################################################################################ ##############################
    ###############################################################################################################
    ########################################################## ####################################################
    ##############################################################################################################
    ### ################################### ######################################################################
    ######################################################################################################################
    ##########
            self.Prepare()
            [strText, retval] = self.Update()
            self.bRunning = False
            return [strText, retval]

        except:
            pass
        finally:
            self.ib.reqGlobalCancel()
            # if self.ib.isConnected():
            #     self.ib.disconnect()
            self.ib.waitOnUpdate(0.5)

        self.bRunning = False
        return ['','']
    # for ML function
    def computeClassification(self, actual):
        if (actual > 0):
            return 1
        else:
            return -1

    # method for coding Black Scholes Call Option Pricing
    def black_scholes_call(self, S, K, T, r, sigma):
        # S: spot price
        # K: strike price

        # T: time to maturity, will vary with time, 1/12 specifies 1 month expiration,
        # roughly what we have when we start on March 20th for April 15th Expiration prices
        # however, we are attempting to predict call price on April 12, roughly 3 days prior to expiration,
        # so this T value will be much less then and cannot remain Fixed
        # for 1 week expiration, T = 7/365, thus for 3 days, we can safely assume 3/365

        # r: interest rate, constant at 1%

        # sigma: volatility of underlying asset

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

        return call

    def get_quote_data(self, symbol, data_range, data_interval):
        res = requests.get(
            'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(
                **locals()))
        data = res.json()
        body = data['chart']['result'][0]
        dt = datetime.datetime
        dt = pd.Series(map(lambda x: arrow.get(x).to('Asia/Calcutta').datetime.replace(tzinfo=None), body['timestamp']),
                       name='Datetime')
        df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        dg = pd.DataFrame(body['timestamp'])
        df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        df.dropna(inplace=True)  # removing NaN rows
        df.columns = ['open', 'high', 'low', 'close', 'volume']  # Renaming columns in pandas

        return df

    # This function runs once at the beginning of the strategy to run initial one-time processes/computations
    def Prepare(self):
        global pricedata_1

        print("Requesting Initial Price Data...")
        pricedata_1 = self.get_quote_data(self.symbol_1, '125d', '1d')
        print("Initial Price Data Received...")

    # Get latest close bar prices and run Update() function every close of bar/candle
    def Run(self):
        while True:
            currenttime = dt.datetime.now()
            if self.timeframe == "1 min" and currenttime.second == 0 and self.GetLatestPriceData():
                self.Update()
            elif self.timeframe == "1 day" and currenttime.second == 0 and currenttime.minute % 5 == 0 and self.GetLatestPriceData():
                self.Update()
                time.sleep(42480)
            elif self.timeframe == "15 mins" and currenttime.second == 0 and currenttime.minute % 15 == 0 and self.GetLatestPriceData():
                self.Update()
                time.sleep(840)
            elif self.timeframe == "1 week" and currenttime.second == 0 and currenttime.minute % 30 == 0 and self.GetLatestPriceData():
                self.Update()
                time.sleep(240)
            elif currenttime.second == 0 and currenttime.minute == 0 and self.GetLatestPriceData():
                self.Update()
                time.sleep(3540)
            time.sleep(1)

    # Returns True when pricedata is properly updated
    def GetLatestPriceData(self):
        global pricedata_1

        # Normal operation will update pricedata on first attempt
        new_pricedata_1 = self.get_quote_data(self.symbol_1, '125d', '1d')
        print(new_pricedata_1)
        return True

        # This function is run every time a candle closes
    def Update(self):
        retval = {}
        strText = ""
        symbols = [self.symbol_1]  # NOTE: *** Enter whichever symbols from lines 65 - 94 that you actually want to test in here
        # INPUT DATA
        for symbol in symbols:
            try:
                ####################################################################################################################################################################################
                # IMPORT DATA
                ####################################################################################################################################################################################

                df = self.get_quote_data(symbol, '125d', '1d')

                ####################################################################################################################
                ####################################################################################################################
                # BLACK SCHOLES OPTIONS PRICING LOGIC
                ####################################################################################################################
                ####################################################################################################################

                # Calculate historical volatility for the stock in question:
                # pct_change_stock = np.log(df['close'] / df['close'].shift(1))
                pct_change_stock = df['close'].pct_change()
                vol_stock_returns = pct_change_stock.std()

                # Annualize vol_stock_returns for historical volatility by multiplying it by square root of 252, if you want monthly volatility multiply by sqrt (21) instead for 21 trading days
                historical_vol = vol_stock_returns * sqrt(252)

                print("Current Annualized Historical Volatility for Stock is:")
                print(historical_vol)
                strText = "{}Current Annualized Historical Volatility for Stock is:\n{}".\
                    format(strText, historical_vol)

                retval["historical_vol"] = historical_vol
                print("BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT")

                bsm_call_price = self.black_scholes_call(
                    df['close'].values[-1:],
                            df['close'].values[-1:] * self.expiration_premium_symbol_1_call,
                            self.time_exp, self.rf_rate, historical_vol)
                print(bsm_call_price)
                strTemp = ','.join([str(item) for item in bsm_call_price])
                strText = "{}BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT\n{}\n".\
                    format(strText, strTemp)
                retval["bsm_call_price"] = bsm_call_price.tolist()

                print("MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                # K - S
                theoretical_call_price = df['close'].values[-1:] * \
                                         self.expiration_premium_symbol_1_call - df['close'].values[-1:]
                print(theoretical_call_price)
                strTemp = ','.join([str(item) for item in theoretical_call_price])
                strText = "{}MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT\n{}\n".\
                    format(strText, strTemp)
                retval["theoretical_call_price"] = theoretical_call_price.tolist()
                # IF BSM CALL PRICE IS GREATER THAN MARKET PREMIUM CALL PRICE, BUY SIGNAL
                # IF BSM CALL PRICE IS CHEAPER THAN MARKET PREMIUM CALL PRICE, SELL SIGNAL

                #########################################################################################################################################################################################
                #########################################################################################################################################################################################
                #########################################################################################################################################################################################
                # AI ML TRADING LOGIC
                #########################################################################################################################################################################################
                #########################################################################################################################################################################################
                #########################################################################################################################################################################################

                final_df = pd.DataFrame(df)

                final_df = final_df.reset_index()

                # del final_df['Datetime']
                final_df = pd.DataFrame(final_df)

                final_df['returns'] = final_df['close'].astype(float).pct_change()

                print("FINAL DF")
                final_df = final_df.fillna(0)
                del final_df['Datetime']
                print(final_df)

                # Compute the last column (Y) -1 = down, 1 = up by applying the defined classifier above to the
                # 'returns_final' dataframe
                final_df.iloc[:, len(final_df.columns) - 1] = final_df.iloc[:, len(final_df.columns) - 1].apply \
                    (self.computeClassification)

                # Now that we have a complete dataset with a predictable value, the last colum “Return”
                # which is either -1 or 1, create the train and test dataset.
                # convert float to int so you can slice the dataframe
                testData = final_df[-int((len(final_df) * 0.10)):]  # forward tested on
                trainData = final_df[:-int((len(final_df) * 0.90))]  # trained on

                # replace all inf with nan
                testData_1 = testData.replace([np.inf, -np.inf], np.nan)
                trainData_1 = trainData.replace([np.inf, -np.inf], np.nan)
                # replace all nans with 0
                testData_2 = testData_1.fillna(0)
                trainData_2 = trainData_1.fillna(0)

                # X is the list of features (Open, High, Low, Close, Volume, StDev, SMA, Upper Bollinger Band, Lower
                # Bollinger Band, RSI, Returns_Final)
                data_X_train = trainData_2.iloc[:, 0:len(trainData_2.columns) - 1]
                # Y is the 1 or -1 value to be predicted (as we added this for the last column above using the apply.
                # (computeClassification) function
                data_Y_train = trainData_2.iloc[:, len(trainData_2.columns) - 1]

                # Same thing for the test dataset
                data_X_test = testData_2.iloc[:, 0:len(testData_2.columns) - 1]
                data_Y_test = testData_2.iloc[:, len(testData_2.columns) - 1]

                logisticregression = LogisticRegression()
                ada = AdaBoostClassifier(base_estimator=logisticregression, n_estimators=100, learning_rate=0.5
                                         , random_state=42)
                # learning rate is a regularization parameter (avoid
                # overfitting), used to minimize loss function, increasing test accuracy.
                clf = BaggingClassifier(base_estimator=ada)
                # learning rate is the contribution of each model to the weights and
                # defaults to 1. Reducing this rate means the weights will be increased or decreased to a small
                # degree, forcing the model to train slower (but sometimes resulting in better performance). n
                # estimators id maximum number of estimators (or models) at which boosting is terminated. In case of
                # perfect fit, the learning procedure is stopped early. It is the maximum number of models to
                # iteratively train. Defaults to 50. random state default = None.  If RandomState instance,
                # random_state is the random number generator; If None, the random number generator is the
                # RandomState instance used by np.random.

                clf.fit(data_X_train, data_Y_train)

                predictions = clf.predict(data_X_test)  # predict y based on x_test
                predictions = pd.DataFrame(predictions)
                print("Accuracy Score Employing Machine Learning: " + str(accuracy_score(data_Y_test, predictions)))
                strText = "{}Accuracy Score Employing Machine Learning:  {}\n".format(strText,
                                str(accuracy_score(data_Y_test, predictions)))
                # BEGIN IB LOGIC:
                #######################################################################################################
                # ##################################################################################
                # CALL AVAILABLE ACCOUNT BALANCE FOR % ALLOCATION INTO PORTFOLIO

                def account_tag_value(ib, tag):
                    return next(a for a in ib.accountSummary() if a.tag == tag).value

                funds = float(account_tag_value(self.ib, 'AvailableFunds'))
                print(f"\n\nAvailableFunds: {funds}")
                strText = "{}\n\nAvailableFunds: {}".format(strText,str(funds))
                # *****************************************************************************************************
                # next, allocate a 2% position based on available balance
                # *****************************************************************************************************
                # CHECK LAST PORTFOLIO BALANCE TO ALLOCATE SHARES BASED ON % OF AVAILABLE BUYING POWER (MARGIN)
                # *****************************************************************************************************
                percent_allocation = 0.01
                print("Current Portfolio Available Balance:")
                strText = strText + "Current Portfolio Available Balance:\n"
                print(funds)
                strText = strText + str(funds)
                qty = (funds * percent_allocation) / final_df['close'].values[-1:]  # latest price of asset
                qty = int(qty)  # set to integer, since its shares, and we want a whole number
                print("Number of Shares to Trade:")
                print(qty)
                strText = "{}Number of Shares to Trade:\n{}\n".format(strText, str(qty))

                print("Current Portfolio Available Balance:")
                print(funds)
                strText = "{}Current Portfolio Available Balance:\n{}\n".format(strText, str(funds))

                current_ai_historical_accuracy = accuracy_score(data_Y_test, predictions)
                current_ai_historical_accuracy = round(current_ai_historical_accuracy, 2)
                print("Current AI historical accuracy:")
                print(current_ai_historical_accuracy)
                strText = "{}Current AI historical accuracy:\n{}\n".format(strText, str(current_ai_historical_accuracy))
                retval["current_ai_historical_accuracy"] = current_ai_historical_accuracy

                # ******* Define what daily drawdown percent on entire portfolio will cause us to stop trading:
                portfolio_stop_loss_percent = 0.03
                portfolio_daily_stop_loss = funds - (funds * portfolio_stop_loss_percent)
                print(portfolio_daily_stop_loss)
                retval["portfolio_daily_stop_loss"] = portfolio_daily_stop_loss

#######################################################################################################################
# ##################################################################
#######################################################################################################################
# ##################################################################
######################################################################################################################
# ####################################################################
#######################################################################################################################
# ###################################################################
#######################################################################################################################
# ###################################################################
#######################################################################################################################
# ###################################################################

                if predictions.values[-1:] == 1 and (current_ai_historical_accuracy > self.ai_pct_threshold_long) and (
                        bsm_call_price > theoretical_call_price):
                    print("BUY LONG SIGNAL!")
                    print("Buying Call Contract for...")
                    print(symbol)

                    # order to buy contract:
                    contract = Option(symbol=symbol, lastTradeDateOrContractMonth=self.expiration_date_call,
                                      strike=self.strike_price_call, right='C', exchange='SMART')
                    # place market order
                    order = MarketOrder('BUY', self.num_contracts)
                    trade = self.ib.placeOrder(contract, order)
                    print(trade)

                if predictions.values[-1:] == -1 and (current_ai_historical_accuracy < self.ai_pct_threshold_short) and (
                        bsm_call_price < theoretical_call_price):
                    print("SELL SHORT SIGNAL!")
                    print("Selling Call Contract for...")
                    print(symbol)

                    # order to sell contract:
                    contract = Option(symbol=symbol, lastTradeDateOrContractMonth=self.expiration_date_put,
                                      strike=self.strike_price_put, right='P', exchange='SMART')
                    # place market order
                    order = MarketOrder('BUY', self.num_contracts)
                    trade = self.ib.placeOrder(contract, order)
                    print(trade)

                # FINALLY, CONFIRM STATUS OF SIGNALS
                print(
                    "***************************************************************************************************")
                print(
                    "***************************************************************************************************")
                print(
                    "***************************************************************************************************")

                print("Latest AI Accuracy for:")
                print(symbol)
                print(current_ai_historical_accuracy)
                strText = "{}Latest AI Accuracy for::\n{}\n{}\n".format(strText,
                        symbol, current_ai_historical_accuracy)

                print("Current Set AI Percent Threshold for Long Side:")
                print(self.ai_pct_threshold_long)
                strText = "{}Current Set AI Percent Threshold for Long Side:\n{}\n".\
                    format(strText, self.ai_pct_threshold_long)

                print("Current Set AI Percent Threshold for Short Side:")
                print(self.ai_pct_threshold_short)
                strText = "{}Current Set AI Percent Threshold for Short Side:\n{}\n".\
                    format(strText, self.ai_pct_threshold_short)

                print("Latest Machine Learning Signal for:")
                print(symbol)
                print(predictions.values[-1:])
                strText = "{}Latest Machine Learning Signal for:\n{}\n{}\n".\
                    format(strText, symbol, ','.join([str(elem) for elem in predictions.values[-1:]]))

                print("BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                print(symbol)

                bsm_call_price = self.black_scholes_call(df['close'].values[-1:], df['close'].values[-1:] * 1.01,
                                                         self.time_exp, self.rf_rate, historical_vol)
                strTemp = ','.join([str(elem) for elem in bsm_call_price])
                strText = "{}BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT\n{}\n{}\n". \
                    format(strText, symbol, strTemp)
                print(bsm_call_price)
                retval["bsm_call_price"] = bsm_call_price.tolist()
                print("MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                print(symbol)
                # K - S
                theoretical_call_price = df['close'].values[-1:] * 1.01 - df['close'].values[-1:]
                print(theoretical_call_price)
                strTemp = ','.join([str(elem) for elem in theoretical_call_price])
                strText = "{}MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT\n{}\n{}\n". \
                    format(strText, symbol, strTemp)
                retval["theoretical_call_price"] = theoretical_call_price.tolist()
                print(str(dt.datetime.now()) + "	 " + self.timeframe + " Update Function Completed.\n")
            except:
                pass
        return [strText, retval]

class getGoogleData:
    def __init__(self):

        pass

def openSample():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

    # The ID and range of a sample spreadsheet.
    SAMPLE_SPREADSHEET_ID = '1-opKcXp7p34ojwWFpM3PP48Fd34s6PCRwLDzmfv7Rts'
    SAMPLE_RANGE_NAME = 'Data!A1:AI'

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    print(os.path.curdir)
    print(settings.CREDENTIAL_PATHS + 'token.pickle')
    if os.path.exists(settings.CREDENTIAL_PATHS + 'token.pickle'):
        with open(settings.CREDENTIAL_PATHS + 'token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.CREDENTIAL_PATHS + 'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(settings.CREDENTIAL_PATHS + 'token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])

    sheets = []
    if not values:
        print('No data found.')
    else:
        print('Name, Major:')

        for row in values[1:]:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (row[0], row[1], row[8], row[9], row[21], row[23], row[25], row[27], row[32], row[33], row[34]))
            sheet = {
                    "symbol_1": row[0],
                    "expiration_call": row[21],
                     "strike_call": row[8],
                     "expiration_put": row[21],
                     "strike_put": row[9],
                     "ten_years_yield": row[23],
                     "time_exp": row[25],
                     "opt_diff": row[27],
                     "num_of_contracts": row[32],
                     "long_threshold": row[33],
                     "short_threshold": row[34]
            }
            sheets.append(sheet)
    return sheets

if __name__ == '__main__':
    openSample()

    # util = Util()
    # sheet = {
    #         "symbol_1" : "AAPL",
    #         "expiration_call": '20200207',
    #          "strike_call": '325',
    #          "expiration_put": '20200207',
    #          "strike_put": '312.50',
    #          "ten_years_yield": '1.84',
    #          "time_exp": '0.049',
    #          "opt_diff": '1.020',
    #          "num_of_contracts": '0',
    #          "long_threshold": '0.58',
    #          "short_threshold": '0.50'}
    # util.getTrade(sheet)
