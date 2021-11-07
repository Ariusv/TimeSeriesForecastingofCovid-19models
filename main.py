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
from training import one_training_step

test_ratio = 0.2
n_steps = 100
epochs = 500
batch_size = 100
scaler = MinMaxScaler()
general_dataframe = GeneralCovidDataFrame()
Country_dataframe = CountryCovidDataFrame(general_dataframe, "Ukraine")



original_target = Country_dataframe.df_for_learning.daily_confirmed_cases
target = np.reshape(np.asarray(original_target), (-1, 1))
normalized_target = scaler.fit_transform(target)

x, y = make_window(normalized_target, n_steps)
actual_data = scaler.inverse_transform(y)


vanilla_lstm = LSTMModel()
vanilla_lstm.load("updated_model_100n.pth")
#score = one_training_step(original_target, scaler, n_steps, test_ratio, vanilla_lstm,  epochs, batch_size)

x= np.reshape(x, (x.shape[0], n_steps, 1))
future_train = x[len(x)-1]
future = []
days_in_future = 30
for i in range(days_in_future):
     future.append(vanilla_lstm.predict(future_train[newaxis,:,:])[0,0])
     future_train = np.insert(future_train, len(x[0]), future[-1], axis=0)
     future_train = future_train[1:]

future = scaler.inverse_transform(np.asarray(future).reshape(-1, 1))


dates = pd.date_range("31-10-2021", periods=days_in_future)
future_data=pd.DataFrame(future, index=dates)
print(original_target.tail())
print(future_data[0])



plt.figure(figsize=(10,6))
plt.plot(original_target, color='blue', label='Actual')
plt.plot(future_data, color='red', label='Predicted data')
plt.title('Test')
plt.xlabel('Date')
plt.ylabel('Confirmed cases')
plt.legend()
plt.show()