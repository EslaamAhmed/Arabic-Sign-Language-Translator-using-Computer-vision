import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional,LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

en_actions = np.array(['What is your name','where is ur home','salam alaykum','im fine','how r u','im tired','need a help','how old r u','ur phone num','im from egypt'])
no_sequences = 50# number of videos for each word
sequence_length= 40 # num of frames in each video

def run_training():
    DATA_PATH='../MP_Data'
    label_map = {label:num for num, label in enumerate(en_actions)}
    #load Data
    sequences, labels = [], []
    for action in en_actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
            
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(40, 258)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(en_actions.shape[0], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('models/bi_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min',verbose=1)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[checkpoint])
    print('Model saved to '+'models/bi_lstm_model.h5')



if __name__ == "__main__":
    run_training()