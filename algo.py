import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# load and prepare the data
def load_and_prepare_data(csv_file, train_split=0.8):
    data = pd.read_csv(csv_file)
    data = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    train_size = int(len(data) * train_split)
    x_train, y_train = [], []
    for i in range(60, train_size):
        x_train.append(data[i-60:i, 0])
        y_train.append(data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_test, y_test = [], data[train_size-60:, 0]
    for i in range(60, len(data)):
        x_test.append(data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

# define the LSTM model
def define_model(train_data):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# function to make a prediction
def predict_price(model, scaler, data, look_back=60):
    new_data = scaler.transform(data.reshape(-1,1))
    x_test = []
    for i in range(look_back, len(new_data)):
        x_test.append(new_data[i-look_back:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

# usage
csv_file = 'historical_stock_prices.csv' # replace with your csv file
x_train, y_train, x_test, y_test, scaler = load_and_prepare_data(csv_file)
model = define_model(x_train)
model.fit(x_train, y_train, epochs=50, batch_size=32)

# predict future prices
future_stock_prices = pd.DataFrame(predict_price(model, scaler, data=x_test[-1:, :, :]), columns=['Close'])
print(future_stock_prices.head())  