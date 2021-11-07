import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import math
from DataFrame import *
from Model import *
from  preprocessing import *
from Model import *
import warnings
from  keras import metrics
from tensorflow import stack

warnings.filterwarnings("ignore")


test_ratio = 0.2
n_units = 150
dropout = 0.2
val_ratio = test_ratio / (1 - test_ratio)
epochs=1000
batch_size=150
days_in_future = 30
scaler = MinMaxScaler()



general_dataframe = GeneralCovidDataFrame()
countries = general_dataframe.confirmed_cases_df["Country/Region"]

data = upload_data_from_file("countries_confirmed_cases.txt")
x=[]
for target in data.values():
    target = np.reshape(np.asarray(target), (-1, 1))
    x.append(np.reshape(scaler.fit_transform(target), ( target.shape[0],)))


encoder_input_data, decoder_target_data = make_window_for_seq2seq(x, days_in_future)
encoder_input_data = reshape_for_seq2seq(encoder_input_data)
decoder_target_data = reshape_for_seq2seq(decoder_target_data)

#teacher_forsing

decoder_input_data = np.zeros(decoder_target_data.shape)
decoder_input_data[:, 1:, 0] = decoder_target_data[:, :-1, 0]
decoder_input_data[:, 0, 0] = encoder_input_data[:, -1, 0]

seq2seq_vanilla_model = EncoderDecoderModel()
seq2seq_vanilla_model.build_vanilla_encoder_decoder_LSTM(n_units=n_units, dropout=dropout)
seq2seq_vanilla_model.compile(lf='mean_squared_error', opt='adam', metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError(), metrics.MeanSquaredLogarithmicError()])
seq2seq_vanilla_model.train([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, val_ratio=val_ratio, cb="without_cb")
seq2seq_vanilla_model.save_loss_plot("s2s")

ukraine_test_data=[]
country_df=CountryCovidDataFrame(general_dataframe, "Ukraine")
target = country_df.df_for_learning.daily_confirmed_cases
target = np.reshape(np.asarray(target), (-1, 1))
ukraine_test_data.append( np.reshape(scaler.fit_transform(target), ( target.shape[0],)))




encoder_input_data, decoder_target_data = make_window_for_seq2seq(ukraine_test_data, days_in_future)
encoder_input_data = encoder_input_data
encoder_input_data = reshape_for_seq2seq(encoder_input_data)

decoder_target_data = decoder_target_data
decoder_target_data = reshape_for_seq2seq(decoder_target_data)

decoder_input_data = np.zeros(decoder_target_data.shape)
decoder_input_data[:, 1:, 0] = decoder_target_data[:, :-1, 0]
decoder_input_data[:, 0, 0] = encoder_input_data[:, -1, 0]

seq2seq_vanilla_model.train([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=10, val_ratio=None, cb="without_cb")
seq2seq_vanilla_model.upgrade_model(n_units)

series = encoder_input_data
prediction = seq2seq_vanilla_model.decode_sequence(series, days_in_future)

series = series.reshape(-1, 1)
prediction = prediction.reshape(-1, 1)

target_series = decoder_target_data.reshape(-1, 1)
target_series = np.concatenate([series[-1:], target_series])

actual_data_tail = series[-len(encoder_input_data[0]):]
x_encode = actual_data_tail.shape[0]

preds = np.zeros((days_in_future + 1, 1))
preds[0] = actual_data_tail[-1]
preds[1:] = prediction

plt.figure(figsize=(10, 6))
plt.plot(range(x_encode, x_encode + days_in_future + 1), target_series, color='green')
plt.plot(range(x_encode, x_encode + days_in_future + 1), preds, color='red', linestyle='--')
plt.plot(range(1, x_encode + 1), actual_data_tail, color='black')
plt.legend(['Actual cases', 'Predictions'])
plt.show()

print(seq2seq_vanilla_model.score([encoder_input_data, decoder_input_data], decoder_target_data))