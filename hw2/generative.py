import sys
import numpy as np
import math 
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
    global mu1,mu2,cnt1,cnt2,sigma1,sigma2,shared_sigma
   	# Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106,106))
    sigma2 = np.zeros((106,106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

def infer(X_test):
    # Predict
    print('Predicting...')
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(cnt1)/cnt2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    return y,y_


load_data(sys.argv[1],sys.argv[2],sys.argv[3])
normalize(X_train,X_test)
train(nor_train,Y_train)
#train(X_train,Y_train)


answer = infer(nor_test)[1]
#answer = infer(X_test)[1]

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
