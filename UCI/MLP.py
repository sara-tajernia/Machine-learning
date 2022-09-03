from sklearn.preprocessing import LabelEncoder
from torch.nn.init import kaiming_uniform_
from torch.utils.data import random_split
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import Sigmoid, CrossEntropyLoss, MSELoss
from torch.nn import BCELoss
from torch.nn import Module
from torch.optim import SGD
from torch.nn import Linear
from torch.nn import ReLU
from numpy import vstack
from torch import Tensor, optim, nn


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, info_list, accept_train):
        # store the inputs and outputs
        self.X = info_list.astype('float32')
        self.y = accept_train
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        # ensure input data is floats
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 24)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(24, 16)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(16, 8)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # fourth hidden layer and output
        self.hidden4 = Linear(8, 1)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
        # self.dropout = nn.Dropout(0.25)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        # X = self.dropout(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        # X = self.dropout(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        # X = self.dropout(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        return X

# prepare the dataset
def prepare_data(info_list, accept_train):
    # load the dataset
    dataset = CSVDataset(info_list, accept_train)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=512, shuffle=True)
    test_dl = DataLoader(test, batch_size=2048, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # print(targets)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(train_dl, test_dl, model):
    predictions_t, actuals = list(), list()
    # print(type(test_dl))
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        # print(2, yhat)
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions_t.append(yhat)
        actuals.append(actual)
    predictions_t, actuals = vstack(predictions_t), vstack(actuals)
    # calculate accuracy
    # acc = accuracy_score(actuals, predictions_t)
    return predictions_t

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return
