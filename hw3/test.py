import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys, csv
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras.callbacks import History

model_path = 'my_model.h5'

test_path = sys.argv[1]
output_dir = sys.argv[2]

img_rows, img_cols = 48, 48
num_classes = 7

def load_data(test_file):
    X_test = []

    k = open(test_file,'r')
    for row in csv.DictReader(k):
        X_test.append(row['feature'])

    X_test = ([row.split(' ')for row in X_test])
    X_test = np.array(X_test)
    X_test = [row.reshape(48, 48) for row in X_test]
    X_test = np.array(X_test)
    X_test = X_test.astype('float32')

    k.close()

    return (X_test)

X_test = load_data(test_path)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')

X_test /= 255

print('Input shape:', input_shape)
print('X_test shape:', X_train.shape)

model = load_model(model_path)

Y_pred = model.predict_classes(X_test)

# Write output
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'prediction.csv')

with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(Y_pred):
        f.write('%d,%d\n' % (i, v))

