import warnings
from keras.models import Sequential
from keras.layers import *
from  keras import metrics
from DataFrame import *
from  preprocessing import *
from Model import *
warnings.filterwarnings("ignore")
from training import *

scaler = MinMaxScaler()
#n_steps = 60
test_ratio = 0.25
#batch_size = n_steps
epochs = 150
#n_units = 100


n_steps_list = [100]
n_units_list = [65, 100]
#countries = ["Germany", "Italy", "Poland", "France", "Spain", "Romania", "Sweden", "Sweden", "Greece"]
countries = ["Germany", "Italy", "Poland", "France", "Spain", "Romania"]

general_dataframe = GeneralCovidDataFrame()
scores = find_parametrsv2(general_dataframe, countries, scaler, test_ratio,epochs,  n_steps_list, n_units_list, max_n_lstm_layers=6)





