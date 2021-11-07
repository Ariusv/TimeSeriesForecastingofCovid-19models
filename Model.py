from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from  keras import callbacks
from keras.models import Model
import numpy as np

class VanillaModel():
    def __init__(self):
        self.history = None
        self.model = None

    def load(self, path):
        self.model = load_model(path)

    def compile(self, lf, opt, metrics):
        self.lf = lf
        self.model.compile(loss=lf, optimizer=opt, metrics=metrics)

    def train(self, x_train, y_train, epochs, batch_size, val_ratio, cb="standart"):
        if cb == "standart":
            cb = [callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=200)]
            self.history = self.model.fit(x_train, y_train, callbacks=cb, epochs=epochs, batch_size=batch_size,
                                         validation_split=val_ratio, shuffle=False)
        elif cb=="without_cb":
            self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                          validation_split=val_ratio, shuffle=False)
        else:
            self.history = self.model.fit(x_train, y_train, callbacks=cb, epochs=epochs, batch_size=batch_size,
                                          validation_split=val_ratio, shuffle=False)
        return self.history

    def score(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        return score

    def save(self, file_path):
        self.model.save(file_path)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

    def save_loss_plot(self, title):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title(title + '. A loss function: ' + self.lf)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(title + ".png")
        plt.clf()


class LSTMModel(VanillaModel):
    def __init__(self):
        super().__init__()

    def build_vanilla_LSTM(self, n_steps, n_units, dropout, n_outputs):
        self.model = Sequential()
        self.model.add(LSTM(n_units, input_shape=(n_steps, 1)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_outputs))

    def build_stacked_LSTM(self, n_layers, n_steps, n_units, dropout, n_outputs):
        self.model = Sequential()
        for _ in range(n_layers-1):
            self.model.add(LSTM(input_shape=(n_steps, 1), return_sequences=True, units=n_units))
        self.model.add(LSTM(input_shape=(n_steps, 1), units=n_units))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(units=n_outputs))



class EncoderDecoderModel(LSTMModel):
    def __init__(self):
        super().__init__()


    def build_vanilla_encoder_decoder_LSTM(self, n_units, dropout):
        self.__encoder_inputs = Input(shape=(None, 1))
        encoder_lstm = LSTM(n_units, dropout=dropout, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(self.__encoder_inputs)
        self.__encoder_states = [state_h, state_c]

        self.__decoder_inputs = Input(shape=(None, 1))

        self.__decoder_lstm = LSTM(n_units, dropout=dropout, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.__decoder_lstm(self.__decoder_inputs, initial_state=self.__encoder_states)

        self.__decoder_dense = Dense(1)
        decoder_outputs = self.__decoder_dense(decoder_outputs)

        self.model = Model([self.__encoder_inputs, self.__decoder_inputs], decoder_outputs)

    def upgrade_model(self, n_units):
        self.__encoder_model = Model(self.__encoder_inputs, self.__encoder_states)

        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.__decoder_lstm(self.__decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = self.__decoder_dense(decoder_outputs)
        self.__decoder_model = Model([self.__decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

    def decode_sequence(self, sequence, n_days):
        states_value = self.__encoder_model.predict(sequence)

        target = np.zeros((1, 1, 1))
        target[0, 0, 0] = sequence[0, -1, 0]

        future = np.zeros((1, n_days, 1))

        for i in range(n_days):

            output, h, c = self.__decoder_model.predict([target] + states_value)
            future[0, i, 0] = output[0, 0, 0]

            target = np.zeros((1, 1, 1))
            target[0, 0, 0] = output[0, 0, 0]
            states_value = [h, c]


        return future


