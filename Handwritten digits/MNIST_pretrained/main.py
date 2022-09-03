from torchvision.models import ResNet50_Weights, resnet50
from torch.utils.data import Dataset
from torchvision import transforms

import torchvision.datasets as dts
import torch.optim as optim
import torch.nn as nn
import torch

trnsform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

mnist_train_set = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
train_loader = torch.utils.data.DataLoader(mnist_train_set, batch_size=64, shuffle=False)

mnist_test_set = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=64, shuffle=False)

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 10, bias=True)
model.eval()

if(torch.cuda.is_available()):
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Training....')
total = 0
correct = 0
for epoch in range(2):
    for i, data in enumerate(train_loader, 1):
        images, labels = data
        if (torch.cuda.is_available()):
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        if (i % 100 == 0):
            print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))

        loss.backward()
        optimizer.step()

print('Training accuracy: {} %'.format((correct / total) * 100))

print('Predicting....')


total = 0
correct = 0
predictions = torch.LongTensor()
for i, data in enumerate(test_loader, 1):

    images, labels = data
    if (torch.cuda.is_available()):
        images = images.cuda()
        labels = labels.cuda()


    if (torch.cuda.is_available()):
        data = data.cuda()

    if (i % 100 == 0):
        print('Batch {} done'.format(i))

    outputs = model(images)
    pred = outputs.cpu().data.max(1, keepdim=True)[1]
    predictions = torch.cat((predictions, pred), dim=0)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Test accuracy: {} %'.format((correct / total) * 100))


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