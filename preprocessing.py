import matplotlib.pyplot as plt
import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
from DataFrame import *
import math

def create_train_valid_test_series(x, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, shuffle=False)
    return x_train, x_val, x_test, y_train, y_val, y_test

def create_train_test_series(x, y, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)
    return x_train, x_test, y_train, y_test

def make_window(sequence, n_steps):
    x, y = [], []
    for i in range(len(sequence) - n_steps):
        x.append(sequence[i: i + n_steps])
        y.append(sequence[i + n_steps])

    return np.array(x), np.array(y)



reshape_for_rnn = lambda x, n_steps: np.reshape(x, (x.shape[0], n_steps, 1))
reshape_for_seq2seq = lambda series: series.reshape((series.shape[0], series.shape[1], 1))

def make_window_for_multistep_LSTM(sequence, n_steps_in, n_steps_out):
    x, y = [],[]
    for i in range(len(sequence)):
        end = i + n_steps_in
        out_end = end + n_steps_out
        if out_end > len(sequence):
            break
        x.append(sequence[i:end])
        y.append(sequence[end:out_end])
    return np.array(x), np.array(y)

def make_window_for_seq2seq(data, n_steps_out):
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][0:-n_steps_out])
        y.append(data[i][-n_steps_out:])
    return np.array(x), np.array(y)

def make_window_for_seq2seq_v2(data, n_steps_in, n_steps_out):
    x = []
    y = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            end = j + n_steps_in
            out_end = end + n_steps_out
            if out_end > len(data[i]):
                break
            x.append(data[i][j:end])
            y.append(data[i][end:out_end])
    return np.array(x), np.array(y)

def make_window_for_seq2seq_v3(data, n_steps_in, n_steps_out):
    x = []
    y = []
    for i in range(len(data)):
        country_data = data[i][14:]
        for j in range(len(country_data)):
            end = j*n_steps_out + n_steps_in
            out_end = end + n_steps_out
            if out_end > len(country_data):
                break
            x.append(country_data[j*n_steps_out:end])
            y.append(country_data[end:out_end])
    return np.array(x), np.array(y)

def save_countries_data(path,general_dataframe, countries, target_country):
    with open(path, "w") as file:
        for country in countries:
            if country != target_country:
                country_df = CountryCovidDataFrame(general_dataframe, country)
                target = country_df.df_for_learning.daily_confirmed_cases

                target_arr = np.array(target)
                file.write(country + ": ")
                for i in range(len(target_arr)):
                    file.write(str(math.fabs(int(target_arr[i]))) + " ")
                file.write("\n")

def upload_data_from_file(path):
    countries_dict = dict()
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = line.split(": ")
            countries_dict[data[0]] = [int(x.split(".")[0]) for x in data[1].split(" ")[:-1]]
    return countries_dict




