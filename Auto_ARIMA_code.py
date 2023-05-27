
############ ARIMA (Auto Regression Integrated Moving Average) Model ##############
import pandas as pd

# Load the dataset
data = pd.read_csv(r"E:\002 project on Forecasting Inventory of Steel Rods\TMT-JUNE-2020_2 (1) - Copy.csv")

# Data Pre-processing

data["Date"] = pd.to_datetime(data["Date"])
data.dtypes

df = data[["Date", "Sales volume in Tonnes"]]

df.isna().sum()

#Handling missing value with Forward Fill method

df = df.fillna(method = 'ffill')
df.isna().sum()

# Plot the time series to visualize any trends or patterns
df['Sales volume in Tonnes'].plot(figsize=(12,5))

# Data Modelling

#pip install pmdarima
from statsmodels.tsa.stattools import adfuller

#Check for Stationarity
def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print("1. ADF: ", dftest[0])
    print("2. P-Value: ", dftest[1])
    print("3. Number of Lags: ", dftest[2])
    print("4. Number of Observations Used for ADF Regression and Critical Values Calculation: ", dftest[3])
    print("5. Critical Values: ")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)
        
ad_test(df['Sales volume in Tonnes'])
# p-value is very small => Stationary

# Figure out Order for ARIMA Model
from pmdarima import auto_arima
#ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")    

stepwise_fit = auto_arima(df['Sales volume in Tonnes'], trace = True, 
                          supress_warnings = True)

stepwise_fit.summary()
# Best Model ARIMA(5,0,5)(0,0,0)=> 000 indicates no Seasonality
#SARIMA is ARIMA Model with Seasonality

from statsmodels.tsa.arima.model import ARIMA

#Split Data into Training and Testing
df.shape
train = df.iloc[:-2000]
test = df.iloc[-2000:]
(train.shape, test.shape)

#Train the Model
model = ARIMA(train['Sales volume in Tonnes'], order = (5,0,5))
model = model.fit()
model.summary()

# Make Predictions on Test Data
start = len(train)
end = len(train)+len(test)-1
pred = model.predict(start = start, end = end, type = 'levels')
pred

#PLot the predicted values
pred.plot()
test['Sales volume in Tonnes'].plot()

#Check for error RMSE
test['Sales volume in Tonnes'].mean()

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(pred, test['Sales volume in Tonnes']))
rmse
norm_rmse = rmse/(df['Sales volume in Tonnes'].max() - df['Sales volume in Tonnes'].min())
norm_rmse
#Mean=8.53 and RMSE=1.5 & Normalised RMSE = 0.13 <1 => error is less compared to Mean => Good Model


model2 = ARIMA(df['Sales volume in Tonnes'], order = (5,0,5))
model2 = model2.fit()
df.tail()

#For future dates
index_future_dates = pd.date_range(start = '2023-02-28', end = '2024-02-28')
index_future_dates

pred = model2.predict(start=len(df), end=len(df)+365,typ='levels').rename("ARIMA Predictions")
pred.index = index_future_dates
pred

pred.plot(figsize = (12,5))

