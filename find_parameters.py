import torch.optim as optim
from torch import nn
from DataFrame import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

from Model import LSTMModel
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential, load_model
from numpy import newaxis
from training import one_training_step, eval_models_for_files
import os
directory = "models"
models_pathes = os.listdir(directory)
import re


test_ratio = 0.2
epochs = 10
batch_size = 100
dropout = 0.2
scaler = MinMaxScaler()
general_dataframe = GeneralCovidDataFrame()
Country_dataframe = CountryCovidDataFrame(general_dataframe, "Ukraine")
original_target = Country_dataframe.df_for_learning.daily_confirmed_cases

#for model_path in models_pathes:
    #params = [int(x) for x in re.findall('[0-9]+', model_path)]
    #n_steps = params[1]
    #n_units = params[2]
    #batch_size = n_units
    #target = np.reshape(np.asarray(original_target), (-1, 1))
    #normalized_target = scaler.fit_transform(target)
    #x, y = make_window(normalized_target, n_steps)
    #actual_data = scaler.inverse_transform(y)
    #lstm = LSTMModel()
    #lstm.load("models\\"+model_path)
    #scores = one_training_step(original_target, scaler, n_steps, test_ratio, lstm,  epochs, batch_size)
    #params_str =model_path + ":" + str(n_steps) + ":" + str(n_units) + ":" + str(scores[0]) + ":" + str(scores[1]) + ":" + str(scores[2]) + ":" + str(scores[3])
    #print(params_str+"\n")
    #with open("optimization.txt", "a") as file:
    #    file.write(params_str+ "\n")
eval_models_for_files(models_pathes, original_target, scaler, test_ratio, epochs)