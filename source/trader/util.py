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

class Util:
    def getSheet(self):
        return

    def getTrade(self, inputArg):
        # Connect to API
        ib = IB()
        ib.connect(settings.TRADER_HOST, settings.TRADER_PORT, clientId=settings.TRADER_CLIENTID)

###### USER PARAMETERS ######
        timeframe = "1 min"
        # Global Variablespip
        pricedata_1 = None
########################################################################################################################
# ##############################################################################
########################################################################################################################
# ##############################################################################
########################################################################################################################
# ##############################################################################
########################################################################################################################
# ##############################################################################
    # INPUTS FOR ALGORITHM:

        symbol_1 = 'AAPL' if inputArg["symbol_1"] is None else inputArg["symbol_1"]
        expiration_date_call = '20200110' if inputArg["expiration_date_call"] is None else inputArg["expiration_date_call"]
        strike_price_call = '310' if inputArg["strike_price_call"] is None else inputArg[
            "strike_price_call"]

        expiration_date_put = '20200110' if inputArg["expiration_date_put"] is None else inputArg[
            "expiration_date_put"]

        strike_price_put = '300' if inputArg["strike_price_put"] is None else inputArg[
            "strike_price_put"]

        # set parameters for risk free rate, and time expiration, for Black Scholes Pricing Model Function:
        rf_rate = 0.017 if inputArg["rf_rate"] is None else inputArg["rf_rate"]
        # risk free rate, normally whatever 10 year treasury yield is

        time_exp = 0.019 if inputArg["time_exp"] is None else inputArg["time_exp"]
        # (30.42 / 365.0) is roughly one month, (7/365) is roughly 1 week, 3/365 is roughly 3 days

        expiration_premium_symbol_1_call = 1.023 if inputArg["expiration_premium_symbol_1_call"] is None else inputArg["expiration_premium_symbol_1_call"]
        # means strike price for the call is approx. 2.3% above current market price ,
        # adjust accordingly to whatever fits your trading parameters profile. So if AAPL was trading
        # at approx 303 at time of these inputs, and our strike_price for call was set to 310, that would be approx.
        # 2.3% above the current market price. Which is why it is set now to 1.023 (2.3% increase
        # from current market price levels when calculated in the BSM model function below,
        # which is why there is a 1 before the .023)

        num_contracts = 1 if inputArg["num_contracts"] is None else inputArg["num_contracts"]
        # specify number of contracts you want to trade

        # set percent thresholds for long and short side needed for ai accuracy on historical data for trade to trigger:
        ai_pct_threshold_long = 0.55 if inputArg["ai_pct_threshold_long"] is None else inputArg["ai_pct_threshold_long"]
        ai_pct_threshold_short = 0.50 if inputArg["ai_pct_threshold_short"] is None else inputArg["ai_pct_threshold_short"]

# ######################################################################################################################
# ############################################################################### ##############################
# ##############################################################################################################
# ######################################################### ####################################################
# #############################################################################################################
# ## ################################### ######################################################################
# #####################################################################################################################

        # for ML function
        def computeClassification(actual):
            if (actual > 0):
                return 1
            else:
                return -1

        # method for coding Black Scholes Call Option Pricing
        def black_scholes_call(S, K, T, r, sigma):
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

        def get_quote_data(symbol, data_range, data_interval):
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
        def Prepare():
            global pricedata_1

            print("Requesting Initial Price Data...")
            pricedata_1 = get_quote_data(symbol_1, '125d', '1d')
            print("Initial Price Data Received...")

        # Get latest close bar prices and run Update() function every close of bar/candle
        def Run():
            while True:
                currenttime = dt.datetime.now()
                if timeframe == "1 min" and currenttime.second == 0 and GetLatestPriceData():
                    Update()
                elif timeframe == "1 day" and currenttime.second == 0 and currenttime.minute % 5 == 0 and GetLatestPriceData():
                    Update()
                    time.sleep(42480)
                elif timeframe == "15 mins" and currenttime.second == 0 and currenttime.minute % 15 == 0 and GetLatestPriceData():
                    Update()
                    time.sleep(840)
                elif timeframe == "1 week" and currenttime.second == 0 and currenttime.minute % 30 == 0 and GetLatestPriceData():
                    Update()
                    time.sleep(240)
                elif currenttime.second == 0 and currenttime.minute == 0 and GetLatestPriceData():
                    Update()
                    time.sleep(3540)
                time.sleep(1)

        # Returns True when pricedata is properly updated
        def GetLatestPriceData():
            global pricedata_1

            # Normal operation will update pricedata on first attempt
            new_pricedata_1 = get_quote_data(symbol_1, '125d', '1d')
            print(new_pricedata_1)

            return True

        # This function is run every time a candle closes
        def Update():

            symbols = [
                symbol_1]  # NOTE: *** Enter whichever symbols from lines 65 - 94 that you actually want to test in here
            # INPUT DATA
            for symbol in symbols:
                try:
                    ####################################################################################################################################################################################
                    # IMPORT DATA
                    ####################################################################################################################################################################################

                    df = get_quote_data(symbol, '125d', '1d')

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

                    print("BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                    bsm_call_price = black_scholes_call(df['close'].values[-1:],
                                                        df['close'].values[-1:] * expiration_premium_symbol_1_call,
                                                        time_exp, rf_rate, historical_vol)
                    print(bsm_call_price)

                    print("MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                    # K - S
                    theoretical_call_price = df['close'].values[-1:] * expiration_premium_symbol_1_call - df[
                                                                                                              'close'].values[
                                                                                                          -1:]
                    print(theoretical_call_price)

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

                    # Compute the last column (Y) -1 = down, 1 = up by applying the defined classifier above to the 'returns_final' dataframe
                    final_df.iloc[:, len(final_df.columns) - 1] = final_df.iloc[:, len(final_df.columns) - 1].apply \
                        (computeClassification)

                    # Now that we have a complete dataset with a predictable value, the last colum “Return” which is either -1 or 1, create the train and test dataset.
                    # convert float to int so you can slice the dataframe
                    testData = final_df[-int((len(final_df) * 0.10)):]  # forward tested on
                    trainData = final_df[:-int((len(final_df) * 0.90))]  # trained on

                    # replace all inf with nan
                    testData_1 = testData.replace([np.inf, -np.inf], np.nan)
                    trainData_1 = trainData.replace([np.inf, -np.inf], np.nan)
                    # replace all nans with 0
                    testData_2 = testData_1.fillna(0)
                    trainData_2 = trainData_1.fillna(0)

                    # X is the list of features (Open, High, Low, Close, Volume, StDev, SMA, Upper Bollinger Band, Lower Bollinger Band, RSI, Returns_Final)
                    data_X_train = trainData_2.iloc[:, 0:len(trainData_2.columns) - 1]
                    # Y is the 1 or -1 value to be predicted (as we added this for the last column above using the apply.(computeClassification) function
                    data_Y_train = trainData_2.iloc[:, len(trainData_2.columns) - 1]

                    # Same thing for the test dataset
                    data_X_test = testData_2.iloc[:, 0:len(testData_2.columns) - 1]
                    data_Y_test = testData_2.iloc[:, len(testData_2.columns) - 1]

                    logisticregression = LogisticRegression()
                    ada = AdaBoostClassifier(base_estimator=logisticregression, n_estimators=100, learning_rate=0.5
                                             ,
                                             random_state=42)  # learning rate is a regularization parameter (avoid overfitting), used to minimize loss function, increasing test accuracy.
                    clf = BaggingClassifier \
                        (
                            base_estimator=ada)  # learning rate is the contribution of each model to the weights and defaults to 1. Reducing this rate means the weights will be increased or decreased to a small degree, forcing the model to train slower (but sometimes resulting in better performance).
                    # n estimators id maximum number of estimators (or models) at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. It is the maximum number of models to iteratively train. Defaults to 50.
                    # random state default = None.  If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

                    clf.fit(data_X_train, data_Y_train)

                    predictions = clf.predict(data_X_test)  # predict y based on x_test
                    predictions = pd.DataFrame(predictions)
                    print("Accuracy Score Employing Machine Learning: " + str(accuracy_score(data_Y_test, predictions)))

                    ########################################################################################################################################################################################
                    ########################################################################################################################################################################################
                    ########################################################################################################################################################################################
                    ########################################################################################################################################################################################
                    # BEGIN IB LOGIC:
                    #########################################################################################################################################################################################
                    # CALL AVAILABLE ACCOUNT BALANCE FOR % ALLOCATION INTO PORTFOLIO

                    def account_tag_value(ib, tag):
                        return next(a for a in ib.accountSummary() if a.tag == tag).value

                    funds = float(account_tag_value(ib, 'AvailableFunds'))
                    print(f"\n\nAvailableFunds: {funds}")

                    # *****************************************************************************************************
                    # next, allocate a 2% position based on available balance
                    # *****************************************************************************************************
                    # CHECK LAST PORTFOLIO BALANCE TO ALLOCATE SHARES BASED ON % OF AVAILABLE BUYING POWER (MARGIN)
                    # *****************************************************************************************************
                    percent_allocation = 0.01
                    print("Current Portfolio Available Balance:")
                    print(funds)

                    qty = (funds * percent_allocation) / final_df['close'].values[-1:]  # latest price of asset
                    qty = int(qty)  # set to integer, since its shares, and we want a whole number
                    print("Number of Shares to Trade:")
                    print(qty)

                    print("Current Portfolio Available Balance:")
                    print(funds)

                    current_ai_historical_accuracy = accuracy_score(data_Y_test, predictions)
                    current_ai_historical_accuracy = round(current_ai_historical_accuracy, 2)
                    print("Current AI historical accuracy:")
                    print(current_ai_historical_accuracy)

                    # ******* Define what daily drawdown percent on entire portfolio will cause us to stop trading:
                    portfolio_stop_loss_percent = 0.03
                    portfolio_daily_stop_loss = funds - (funds * portfolio_stop_loss_percent)
                    print(portfolio_daily_stop_loss)

                    #########################################################################################################################################################################################
                    #########################################################################################################################################################################################
                    ##########################################################################################################################################################################################
                    ##########################################################################################################################################################################################
                    ##########################################################################################################################################################################################

                    ##########################################################################################################################################################################################

                    if predictions.values[-1:] == 1 and (current_ai_historical_accuracy > ai_pct_threshold_long) and (
                            bsm_call_price > theoretical_call_price):
                        print("BUY LONG SIGNAL!")
                        print("Buying Call Contract for...")
                        print(symbol)

                        # order to buy contract:
                        contract = Option(symbol=symbol, lastTradeDateOrContractMonth=expiration_date_call,
                                          strike=strike_price_call, right='C', exchange='SMART')
                        # place market order
                        order = MarketOrder('BUY', num_contracts)
                        trade = ib.placeOrder(contract, order)
                        print(trade)

                    if predictions.values[-1:] == -1 and (current_ai_historical_accuracy < ai_pct_threshold_short) and (
                            bsm_call_price < theoretical_call_price):
                        print("SELL SHORT SIGNAL!")
                        print("Selling Call Contract for...")
                        print(symbol)

                        # order to sell contract:
                        contract = Option(symbol=symbol, lastTradeDateOrContractMonth=expiration_date_put,
                                          strike=strike_price_put, right='P', exchange='SMART')
                        # place market order
                        order = MarketOrder('BUY', num_contracts)
                        trade = ib.placeOrder(contract, order)
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

                    print("Current Set AI Percent Threshold for Long Side:")
                    print(ai_pct_threshold_long)

                    print("Current Set AI Percent Threshold for Short Side:")
                    print(ai_pct_threshold_short)

                    print("Latest Machine Learning Signal for:")
                    print(symbol)
                    print(predictions.values[-1:])

                    print("BSM THEORETICAL CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                    print(symbol)
                    bsm_call_price = black_scholes_call(df['close'].values[-1:], df['close'].values[-1:] * 1.01, time_exp,
                                                        rf_rate, historical_vol)
                    print(bsm_call_price)

                    print("MARKET PREMIUM FOR CALL PRICE WITH STRIKE INCREASE OF N PERCENT")
                    print(symbol)
                    # K - S
                    theoretical_call_price = df['close'].values[-1:] * 1.01 - df['close'].values[-1:]
                    print(theoretical_call_price)

                    print(str(dt.datetime.now()) + "	 " + timeframe + " Update Function Completed.\n")


                except:
                    pass

# def main():
#     Prepare()  # Initialize strategy
#     Run()  # Run strategy
#
# if __name__ == '__main__':
#     main()