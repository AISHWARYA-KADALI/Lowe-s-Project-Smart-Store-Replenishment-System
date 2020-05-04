import warnings
import itertools
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

conn=mysql.connector.connect(host='localhost',database='hackathon',user='root')
if conn.is_connected():
    print('Connected to Mysql Database')
cursor=conn.cursor()

def Demand_Forecasting():
    df = pd.read_excel("Sample - Superstore.xls")
    df_furni = df.loc[df['Category'] == 'Furniture']
    df.head()
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    df_furni.drop(cols, axis=1, inplace=True)
    df_furni = df_furni.sort_values('Order Date')
    df_furni.isnull().sum()
    df_furni = df_furni.set_index('Order Date')
    df_furni.index
    y = df_furni['Sales'].resample('MS').mean()
    y.plot(figsize=(15, 6))
    plt.show()
    #Seasonal decomposition using moving averages.
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()
    #ARIMA MODEL
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations possible for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    #Fitting the ARIMA model
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    #Validating Forecasts
    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('df_furni Sales')
    plt.legend()
    plt.show()
    y_forecasted = pred.predicted_mean
    y_truth = y['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    
    
    #Producing and visualizing forecasts
    pred_uc = results.get_forecast(steps=50)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='g', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('df_furni Sales')
    plt.legend()
    plt.show()
    
    print(pred_ci)
    rowData = pred_ci.loc[ '2020-04-01' , : ]
    print(rowData)
    res=(rowData[0]+rowData[1])/2
    print(res)

    min_val=100
    query="insert into QNTY_MINI(minimum_value,order_quantity) values(%s,%s)"
    values=(min_val,res)
    cursor.execute(query,values)
    conn.commit()
    print("inserted")

Demand_Forecasting()