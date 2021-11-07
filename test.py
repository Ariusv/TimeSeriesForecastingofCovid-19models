import pandas as pd
from training import *
n_steps_list=[2, 4, 8, 10, 20, 40, 60,  100]
n_units_list=[2, 4, 8, 10, 20, 40, 60, 100]
batch_size_list=[2, 4, 8, 10, 20, 40, 60, 100]
epochs_list=[200, 400]
test_ratio_list = [0.25]
dropouts_list = [0.2]
countries = ["Ukraine"]
scaler = MinMaxScaler()

find_parametrsv1(n_steps_list, batch_size_list, test_ratio_list, epochs_list, n_units_list, dropouts_list, countries, scaler)