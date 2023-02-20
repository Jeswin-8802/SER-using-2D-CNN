import os
import numpy as np
from numpy import load
import joblib
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category = DeprecationWarning)

def split_and_expand_dim(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=125, test_size=0.25, shuffle=True)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    return x_train, y_train, x_test, y_test

def get_data():
    X, Y = [], []
    for i in range(5):
        dict_data = load(os.path.join(os.path.abspath('..'), 'data', 'data_x_' + str(part + 1) +  '.npz'))
        X = X.append(dict_data['arr_0'])
        dict_data = load(os.path.join(os.path.abspath('..'), 'data', 'data_y_' + str(part + 1) +  '.npz'))
        Y = Y.append(dict_data['arr_0'])
    print('X.shape, Y.shape ==> ', (X.shape, Y.shape))
    return X, Y

def one_hot_encode(Y):
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    return Y, encoder

def construct_model():
    model = Sequential()
    model.add(MaxPooling2D((2, 2), input_shape = (106, 160, 1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def fit_model(model, x_train, y_train, x_test, y_test):
    epochs = 100
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.75,
                         verbose=0, patience=2, min_lr=0.000001)
    history = model.fit(x_train, y_train, batch_size=25, epochs=epochs,
                    validation_data=(x_test, y_test), callbacks=[rlrp])
    model_store_path = os.path.join(os.path.abspath('..'), 'data', 'lstm.pkl')
    joblib.dump(history, model_store_path)

def main():
    X, Y = get_data()
    Y, encoder = one_hot_encode(Y)
    x_train, y_train, x_test, y_test = split_and_expand_dim(X, Y)
    model = construct_model()
    fit_model(model, x_train, y_train, x_test, y_test)
    print("Accuracy of our model on test data : ", model.evaluate(x_test, y_test)[1] * 100, "%")
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)
    y_test = encoder.inverse_transform(y_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

