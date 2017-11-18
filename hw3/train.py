import numpy as np
import os, sys, csv
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import History 

batch_size = 256
num_classes = 7
epochs = 200

# input image dimensions
img_rows, img_cols = 48, 48

train_path = './data/train.csv'
test_path = './data/test.csv'
output_path ='./data/output'

model_path = 'my_model.h5'

def load_data(train_file, test_file):
    X_train, Y_train, X_test = [],[],[] 

    f = open(train_file, 'r')
    for row in csv.DictReader(f):
        X_train.append(row['feature'])
        Y_train.append(row['label'])
    
    k = open(test_file,'r')
    for row in csv.DictReader(k):
        X_test.append(row['feature'])

    X_train = ([row.split(' ') for row in X_train])
    X_train = np.array(X_train) 
    X_train = [row.reshape(48, 48) for row in X_train]
    X_train = np.array(X_train) 
    X_train = X_train.astype('float32')
    X_test = ([row.split(' ')for row in X_test])
    X_test = np.array(X_test) 
    X_test = [row.reshape(48, 48) for row in X_test]
    X_test = np.array(X_test)
    X_test = X_test.astype('float32')

    f.close()
    k.close()

    return (X_train, Y_train, X_test)
    
X_train, Y_train, X_test = load_data(train_path,test_path)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, shuffle=True)

print('Input shape:', input_shape)
print('X_train shape:', X_train.shape)

# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_valid = keras.utils.to_categorical(Y_valid, num_classes)

# build model
model = Sequential()

# CNN
# block 1
model.add(Conv2D(32, (5, 5), padding='valid', activation='relu',input_shape=input_shape))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Dropout(0.3))

# block 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.3))

# block 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.3))

# block 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

# DNN
model.add(Dense(512, activation='relu',kernel_initializer="uniform", use_bias=True ))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu',kernel_initializer="uniform", use_bias=True ))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu',kernel_initializer="uniform", use_bias=True ))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adadelta(lr = 0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
model.summary()

score = model.evaluate(X_valid, Y_valid, verbose=0)

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, Y_valid), shuffle=True)

print(history.history.keys())
print(history.history['val_acc'])
print(history.history['acc'])

print('Test loss:', score[0])
print('Test accuracy:', score[1]*100,'%')

Y_pred = model.predict_classes(X_test)

# Write output
output_dir = '.'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'result_'+str(batch_size)+'_'+str(epochs)+'.csv')


with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(Y_pred):
        f.write('%d,%d\n' % (i, v))

model.save(model_path)

# emotion_classifier = load_model(model_path)
# plot_model(emotion_classifier,to_file='./model.png')

