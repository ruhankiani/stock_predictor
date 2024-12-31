import os
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
update = 1

def get_stock_data(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Returns dataFrame containing stock data for the given symbol from the Alpha Vantage API.

    Preconditions:
    - symbol must be a valid stock symbol in the Alpha Vantage API
    - api_key must be a valid Alpha Vantage API key

    """

    file_path = f'data/{symbol}.csv'

    if not os.path.exists(file_path):
        df = pdr.DataReader(symbol, "av-daily", api_key=api_key)
        df.to_csv(file_path, index = False)
    else:
        update = input("Do you want to resue previous data or request for updated data? (1/2)\n"\
                        "1. Use previous data \n"\
                        "2. Request updated data \n")

        if update == 2:
            df = pdr.DataReader(symbol, "av-daily", api_key=api_key)
            df.to_csv(file_path, index = False)
        else:
            df = pd.read_csv(file_path)

    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    return df

def prepare_training_data(data_training: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares training data for the model. Returning a tuple containing the training input data (x_train),
    training labels (y_train).

    """

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []


    for i in range(100,data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

    return np.array(x_train), np.array(y_train)

def prepare_testing_data(data_testing: pd.Series, data_training: pd.Series) -> tuple:
    """
    Prepares testing data for the model by combining the last 100 days of training data
    with the testing data, normalizing the combined dataset, and returns a tuple of testing
    input data (x_test) and testing labels (y_test).
    """

    past_100_days = data_training.tail(100)
    final_df = past_100_days._append(data_testing, ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range (100,  input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    return np.array(x_test), np.array(y_test)


company = input("Enter the stock symbol (e.g., AAPL for Apple): ").strip().upper()

API_KEY = # Enter alpha vantage api key https://www.alphavantage.co/support/#api-key

df = get_stock_data(company, API_KEY)

data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])

x_train, y_train = prepare_training_data(data_training)
x_test, y_test = prepare_testing_data(data_testing, data_training)



