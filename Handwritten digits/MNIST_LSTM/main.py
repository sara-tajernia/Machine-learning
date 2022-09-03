from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as  tf

if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = Sequential()
    model.add(LSTM(32, activation='relu', kernel_initializer='he_uniform', input_shape=(x_train.shape[1:])))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    scores, histories = list(), list()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('accuracy= ', acc)
    scores.append(acc)
    histories.append(history)
