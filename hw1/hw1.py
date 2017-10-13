import csv 
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
from numpy.linalg import inv


data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('./data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")


for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != 'NR':
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1

text.close()

avg = []
sd = []
nor_data = []

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

for row in range(0,len(data)):
    nor_data.append(get_normalized_list(data[row])[0])
    avg.append(get_normalized_list(data[row])[1])
    sd.append(get_normalized_list(data[row])[2])


top_i_data = []
rank_index = []

# caculate correlation coefficient
corr = []
for i in range(len(data)):
    corr.append(np.corrcoef(nor_data[9],nor_data[i])[0][1])

def top_i_index(x):
    sorted_corr = sorted(corr)
    for i in range(0,x):
        rank_index.append((corr.index(sorted_corr[-i-1])))
        top_i_data.append(nor_data[rank_index[i]])

top_i_index(5)


# x = []
# y = []
# # 每 12 個月
# for i in range(12):
#     # 一個月取連續10小時的data可以有471筆
#     for j in range(471):
#         x.append([])
#         # top_N種污染物
#         for t in range(int(sys.argv[1])):
#             # 連續9小時
#             for s in range(9):
#                 x[471*i+j].append(top_i_data[t][480*i+j+s])
#         y.append(nor_data[9][480*i+j+9])

# x = np.array(x)
# y = np.array(y)

# # add square term
# x = np.concatenate((x,x**2), axis=1)

# # add bias
# x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

# w = np.zeros(len(x[0]))

# learning rate & iteration & lamda
# l_rate = 100
# repeat = 30000
# lamda = 0.1

# x_t = x.transpose()
# s_gra = np.zeros(len(x[0]))
# regular = np.zeros(len(x))

# training
# for i in range(repeat):
#     hypo = np.dot(x,w)
#     loss = hypo - y 
#     cost = (np.sum(loss**2) + lamda*np.sum(w**2)) / len(x)
#     cost_a  = math.sqrt(cost)
#     gra = 2*(np.dot(x_t,loss) + lamda*w)
#     s_gra += gra**2
#     ada = np.sqrt(s_gra)
#     w = w - l_rate * gra/ada

#     print ('Iteration: %d | Cost: %f' % ( i,cost_a))


# save model
# np.save('./model.npy',w)
# read model
w = np.load('./model.npy')

test_x = []
test_data = []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text , delimiter= ",")

#testing
for r in row:
    test_x.append([])
    for i in range(2,11):
        if r[i] !="NR" :
            test_x[n_row].append(float(r[i]))
        else:
            test_x[n_row].append(0)
    n_row = n_row+1


for r in range(len(test_x)):
    for hour in range(len(test_x[r])):
        test_x[r][hour] = (test_x[r][hour] - avg[r%18]) / sd[r%18]


for day_shift in range(0, len(test_x), 18):
    tmp = []
    for index in rank_index:
        tmp += test_x[day_shift + index]
    test_data.append(tmp)

text.close()
test_data = np.array(test_data)

# add square term
test_data = np.concatenate((test_data,test_data**2), axis=1)

# add bias
test_data = np.concatenate((np.ones((test_data.shape[0],1)),test_data), axis=1)

ans = []
for i in range(len(test_data)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_data[i]) * sd[9] + avg[9]
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()