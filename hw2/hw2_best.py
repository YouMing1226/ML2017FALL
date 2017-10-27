import csv
import numpy as np
import sys
import math

############ MODEL OPTION ############
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import time
st = time.time()

############ READ DATA ############
with open(sys.argv[1], 'rt') as f:
    reader = csv.reader(f)
    X_train = list(reader)[1:]

X_train = [list(map(float, row)) for row in X_train]

with open(sys.argv[2], 'rt') as f2:
    reader = csv.reader(f2)
    y_train = list(reader)
    y_train = [ i for a in y_train for i in a][1:]

y_train = list(map(int, y_train))

with open(sys.argv[3], 'rt') as f3:
    reader = csv.reader(f3)
    X_test = list(reader)[1:]

X_test = [list(map(float, row)) for row in X_test]

# print(X_train[3])
# print(y_train[3])
# print(X_test[3])

# get average
def get_avg(list):
    return sum(list)/len(list)

# get standard deviation
def get_sd(list):
    avg = get_avg(list)
    sum_of_square = 0
    for e in list:
        sum_of_square += (e-avg)**2
    return math.sqrt(sum_of_square/len(list))

# get (normalized list, average, standard deviation)
def get_normalized_list(list):
    avg = get_avg(list)
    sd = get_sd(list)
    normalized_list = [ (e-avg)/sd for e in list]
    return (normalized_list, avg, sd)

print('Start normalize data')
n_train = len(X_train)
n_test = len(X_test)
print('> Training data: ' + str(n_train))
print('> Testing data: ' + str(n_test))

M = X_train+X_test
M = np.array(M)
M_T = M.T
X_all_T = np.array([ get_normalized_list(list(row))[0] for row in M_T])
X_all = X_all_T.T

X_train_norm = X_all[:n_train]
X_test_norm = X_all[n_train:]


############ TRAINING ############
print('Start training')
tree_num = 1000
clf = GradientBoostingClassifier(n_estimators=tree_num, learning_rate=1.0, random_state=0)
clf.fit(X_train_norm, y_train)

############ PREDICTING ############
print('Start predicting')
predict = clf.predict(X_test_norm)

predict_add_index = []
for i in range(len(predict)):
    predict_add_index.append([i + 1])
    predict_add_index[i].append(predict[i])

############ SAVING RESULT ############
filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(['id', 'label'])
for i in range(len(predict_add_index)):
    s.writerow(predict_add_index[i])
text.close()
print('prediction file saved')

et = time.time()
print('Total time: ' + str(et - st) + 's used')
