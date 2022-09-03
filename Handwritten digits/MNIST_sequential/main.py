from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import Sequential
import numpy as np


class MNIST():
    def __init__(self):
        (self.trainX, self.trainy), (self.testX, self.testy) = mnist.load_data()
        self.model = Sequential()

    def normalize(self):
        # flattening
        self.trainX = self.trainX.reshape(60000, 784)
        self.testX = self.testX.reshape(10000, 784)
        self.trainX = np.true_divide(self.trainX, 255)
        self.testX = np.true_divide(self.testX, 255)

    def labels(self):
        trainyb = np.zeros((len(self.trainy), 10))
        testyb = np.zeros((len(self.testy), 10))
        for i in range(len(self.trainy)):
            trainyb[i][self.trainy[i]] = 1
        for i in range(len(self.testy)):
            testyb[i][self.testy[i]] = 1
        self.trainy = trainyb
        self.testy = testyb

    def create_model(self):
        self.model.add(Dense(512, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.trainX, self.trainy, epochs=5, batch_size=64)

    def predict_model(self):
        score = self.model.evaluate(self.testX, self.testy)
        y_pre = self.model.predict(self.testX)
        prey = unique_digits(y_pre)
        testy = unique_digits(self.testy)
        print(classification_report(testy, prey))
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


def show_image(id):
    plt.matshow(id, cmap='gray')
    plt.colorbar()
    plt.show()


def unique_digits(y):
    ys = []
    for p in y:
        max_item = max(p)
        index_list = [index for index in range(len(p)) if p[index] == max_item]
        ys.append(index_list[0])
    return ys


if __name__ == '__main__':
    MNIST = MNIST()
    MNIST.normalize()
    MNIST.labels()
    MNIST.create_model()
    MNIST.train_model()
    MNIST.predict_model()





































# from keras.layers import Dense, Activation, Dropout
# from sklearn.metrics import classification_report
# from tensorflow.keras.datasets import mnist
# from matplotlib import pyplot as plt
# from keras import Sequential
# import numpy as np
#
#
# class MNIST():
# 	def __init__(self):
# 		(self.trainX, self.trainy), (self.testX, self.testy) = mnist.load_data()
# 		self.model = Sequential()
#
#
# 	def normalize(self):
# 		#flattening
# 		self.trainX = self.trainX.reshape(60000, 784)
# 		self.testX = self.testX.reshape(10000, 784)
# 		self.trainX = np.true_divide(self.trainX, 255)
# 		self.testX = np.true_divide(self.testX, 255)
#
#
# 	def labels(self):
# 		trainyb = np.zeros((len(self.trainy), 10))
# 		testyb = np.zeros((len(self.testy), 10))
# 		for i in range(len(self.trainy)):
# 			trainyb[i][self.trainy[i]] = 1
# 		for i in range(len(self.testy)):
# 			testyb[i][self.testy[i]] = 1
# 		self.trainy = trainyb
# 		self.testy = testyb
#
#
# 	def create_model(self):
# 		self.model.add(Dense(512, input_shape=(784,)))
# 		self.model.add(Activation('relu'))
# 		self.model.add(Dropout(0.2))
#
# 		self.model.add(Dense(512))
# 		self.model.add(Activation('relu'))
# 		self.model.add(Dropout(0.2))
#
# 		self.model.add(Dense(10))
# 		self.model.add(Activation('softmax'))
# 		self.model.summary()
#
# 		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#
# 	def train_model(self):
# 		self.model.fit(self.trainX, self.trainy, epochs=5, batch_size=64)
#
#
# 	def predict_model(self):
# 		score = self.model.evaluate(self.testX, self.testy)
# 		y_pre = self.model.predict(self.testX)
# 		prey = unique_digits(y_pre)
# 		testy = unique_digits(self.testy)
# 		print(classification_report(testy, prey))
# 		print('Test score:', score[0])
# 		print('Test accuracy:', score[1])
#
#
# def show_image(id):
# 	plt.matshow(id, cmap='gray')
# 	plt.colorbar()
# 	plt.show()
#
#
# def unique_digits(y):
# 	ys = []
# 	for p in y:
# 		max_item = max(p)
# 		index_list = [index for index in range(len(p)) if p[index] == max_item]
# 		ys.append(index_list[0])
# 	return ys
#
# if __name__ == '__main__':
# 	MNIST = MNIST()
# 	MNIST.normalize()
# 	MNIST.labels()
# 	MNIST.create_model()
# 	MNIST.train_model()
# 	MNIST.predict_model()
