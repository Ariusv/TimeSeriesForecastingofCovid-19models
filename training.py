import warnings
from keras.models import Sequential
from keras.layers import *
from  keras import metrics
from DataFrame import *
from  preprocessing import *
from Model import *
import re
warnings.filterwarnings("ignore")


def training(general_dataframe, countries, scaler,n_steps, test_ratio, model, epochs, batch_size, title):
    scores=[]
    for country in countries:
        country_df=CountryCovidDataFrame(general_dataframe, country)
        target = country_df.df_for_learning.daily_confirmed_cases
        score = one_training_step(target, scaler,n_steps, test_ratio, model,  epochs, batch_size)
        model.save_loss_plot(title+country)
        model.save(title+country+".pth")
        scores.append(score)

    return scores

def one_training_step(target, scaler,n_steps, test_ratio, model,  epochs, batch_size):
    val_ratio = test_ratio / (1 - test_ratio)

    target = np.reshape(np.asarray(target), (-1, 1))
    normalized_target = scaler.fit_transform(target)
    x, y = make_window(normalized_target, n_steps)
    x_train, x_test, y_train, y_test = create_train_test_series(x, y, test_ratio)

    x_train = reshape_for_rnn(x_train, n_steps)
    x_test = reshape_for_rnn(x_test, n_steps)

    model.train(x_train, y_train, epochs=epochs, batch_size=batch_size, val_ratio=val_ratio)
    score = model.score(x_test, y_test)
    return score

def find_parametrsv1(n_steps_list, batch_size_list, test_ratio_list, epochs_list, n_units_list, dropouts_list, countries, scaler):
    for n_units in n_units_list:
        for n_steps in n_steps_list:
            for batch_size in batch_size_list:
                for test_ratio in test_ratio_list:
                    for dropout in dropouts_list:
                        for epochs in epochs_list:
                            vanilla_lstm = LSTMModel()
                            vanilla_lstm.build_vanilla_LSTM(n_steps=n_steps, n_units=n_units, dropout=dropout, n_outputs=1)
                            vanilla_lstm.compile(lf='mean_squared_error', opt='adam', metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError(), metrics.MeanSquaredLogarithmicError()])
                            scores = training(countries, scaler, n_steps, test_ratio, vanilla_lstm, epochs, batch_size)
                            params_str="units: "+str(n_units)+", steps: "+str(n_steps)+", batch_size: "+str(batch_size)+", test_ratio: "+str(test_ratio)+", dropout: "+str(dropout)+" "
                            scores_str = 'Scores: MSE: '+str(scores[0][0])+" RMSE: "+str(scores[0][1])+" MAE: "+str(scores[0][2])+" RMSLE: "+str(scores[0][3])
                            print(params_str+scores_str+"\n")
                            with open("optimization.txt", "a") as file:
                                file.write(params_str+scores_str+"\n")

def find_parametrsv2(general_dataframe, countries, scaler, test_ratio,epochs,  n_steps_list, n_units_list, max_n_lstm_layers):
    scores = []
    for n_steps_ in n_steps_list:
        for n_units_ in n_units_list:
            for n_lstm_layers in range(1, max_n_lstm_layers):
                title = f"stacked{n_lstm_layers}_model_{n_steps_}steps_{n_units_}layers"
                stacked_lstm = LSTMModel()
                stacked_lstm.build_stacked_LSTM(n_layers=n_lstm_layers, n_steps=n_steps_, n_units=n_units_, dropout=0.2,
                                                n_outputs=1)
                stacked_lstm.compile(lf='mean_squared_error', opt='adam',
                                     metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError(),
                                              metrics.MeanSquaredLogarithmicError()])
                score = training(general_dataframe, countries, scaler, n_steps_, test_ratio, stacked_lstm,
                                         epochs, batch_size=n_units_, title=title)
                scores.append(score)
    return scores


def eval_models_for_files(models_pathes, original_target, scaler, test_ratio, epochs):
    for model_path in models_pathes:
        params = [int(x) for x in re.findall('[0-9]+', model_path)]
        n_steps = params[1]
        n_units = params[2]
        batch_size = n_units
        target = np.reshape(np.asarray(original_target), (-1, 1))
        normalized_target = scaler.fit_transform(target)
        x, y = make_window(normalized_target, n_steps)
        actual_data = scaler.inverse_transform(y)
        lstm = LSTMModel()
        lstm.load("models\\" + model_path)
        scores = one_training_step(original_target, scaler, n_steps, test_ratio, lstm, epochs, batch_size)
        params_str = model_path + ":" + str(n_steps) + ":" + str(n_units) + ":" + str(scores[0]) + ":" + str(
            scores[1]) + ":" + str(scores[2]) + ":" + str(scores[3])
        print(params_str + "\n")
        with open("optimization.txt", "a") as file:
            file.write(params_str + "\n")

