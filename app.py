import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import pandas_datareader as data
import streamlit as st
from keras.models import load_model

start='2010-01-01'
end='2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df=yf.download(user_input, start,end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



model= load_model('my_model.keras')


past_100_days = data_training.tail(100)
final_data=pd.concat([past_100_days,data_testing], ignore_index=True)
input_Data=scaler.fit_transform(final_data)

x_test=[]
y_test=[]

for i in range(100, input_Data.shape[0]):
  x_test.append(input_Data[i-100:i])
  y_test.append(input_Data[i,0])

x_test_test, y_test=np.array(x_test), np.array(y_test)

y_predict=model.predict(x_test)
scalar = scaler.scale_

scale_factor=1/scalar[0]
y_predict=y_predict*scale_factor
y_test=y_test*scale_factor

st.subheader('Predictions vs Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Orginal Price')
plt.plot(y_predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)