import sys
import numpy as np
from math import log, floor
import pandas as pd
import csv

# load data
def load_data(train_data_path, train_label_path, test_data_path):
    global X_train,X_test,Y_train
    print('Loading...')
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

 
# normalize
def normalize(X_train, X_test):
    global nor_train, nor_test

    print('Normalizing...')
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_train, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    nor_train = X_train_test_normed[0:X_train.shape[0]]
    nor_test = X_train_test_normed[X_train.shape[0]:]

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def train(X_train, Y_train):
    print('Training...')
    global w,b,z
    # Initiallize parameter, hyperparameter
    w = np.zeros((106,))
    b = np.zeros((1,))
    l_rate = 0.1
    batch_size = 64
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1000

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            z = np.dot(X, np.transpose(w)) + b
            z = np.clip(z,-709,709)
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.sum(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad


def infer(X_test):
    print('Predicting...')
    test_data_size = len(X_test)
    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    z = np.clip(z,-709,709)
    y = sigmoid(z)
    y_ = np.around(y)

    return y,y_



load_data(sys.argv[1],sys.argv[2],sys.argv[3])
normalize(X_train,X_test)
train(X_train,Y_train)

answer = infer(X_test)[1]

print('Saving...')
predict_add_index =[]
for i in range(len(answer)):
    predict_add_index.append([i+1])
    predict_add_index[i].append(int(answer[i]))

filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(predict_add_index)):
    s.writerow(predict_add_index[i]) 
text.close()









