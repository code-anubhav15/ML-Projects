import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model
model = load_model('Stock_Prediction_Model.h5')

# Streamlit UI setup
st.header('Stock Price Predictor')

# User inputs for stock symbol and date range
stock = st.text_input("Enter Stock Symbol", 'GOOG')
start = st.text_input("Enter Start Date", '2010-01-01')
end = st.text_input("Enter End Date", '2024-05-31')

# Download stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Data preparation
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test)

# Moving Averages Plot
st.subheader('Price vs MA')
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data.index, data['Close'], 'g', label='Close Price')
ax1.plot(data.index, ma_100_days, 'b', label='100-Day MA')
ax1.plot(data.index, ma_200_days, 'r', label='200-Day MA')
ax1.set_xlabel('Date')
plt.xticks(rotation=45)
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

# Prepare test data for prediction
x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i - 100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting
predict = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
predict = predict * scale_factor
y_test = y_test * scale_factor

# Original Price vs Predicted Price Plot
st.subheader('Original Price vs Predicted Price')
fig2, ax2 = plt.subplots(figsize=(8, 6))
date_range = data.index[-len(y_test):]

ax2.plot(date_range, y_test, 'g', label='Original Price')
ax2.plot(date_range, predict, 'r', label='Predicted Price')
ax2.set_xlabel('Date')
plt.xticks(rotation=45)
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

# # Future price prediction
# st.subheader('Future Price Prediction')
# future_days = st.slider('Select number of months to predict', 1, 20)
# future_days = future_days * 30

# last_100_days = data_test_scale[-100:]
# future_predictions = []

# for _ in range(future_days):
#     future_pred = model.predict(last_100_days.reshape(1, 100, 1))
#     future_pred = future_pred[0][0] * scale_factor
#     future_predictions.append(future_pred)
#     new_entry = np.append(last_100_days[1:], [[future_pred / scale_factor]], axis=0)
#     last_100_days = new_entry

# # Future dates
# last_date = data.index[-1]
# future_dates = pd.date_range(start=last_date, periods=future_days + 1).tolist()[1:]

# # Combine last 5 days of actual data with future predictions
# recent_5_days = data['Close'][-5:].values.tolist()
# combined_data = recent_5_days + future_predictions
# combined_dates = data.index[-5:].tolist() + future_dates

# # Future Price Plot
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# ax3.plot(combined_dates[:5], recent_5_days, 'g', label='Last 5 Days Actual Price')
# ax3.plot(combined_dates[4:], combined_data[4:], 'r', label='Predicted Future Price')
# ax3.set_xlabel('Date')
# plt.xticks(rotation=45)
# ax3.set_ylabel('Price')
# ax3.legend()
# st.pyplot(fig3)

# Display future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
st.write(future_df)
