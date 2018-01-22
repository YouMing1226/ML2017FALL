import matplotlib
matplotlib.use('Agg')
from skimage import io
import numpy as np 
import os, sys
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
### parameters
train_ratio = 0.9
epochs = 150
batch_size = 256

img_path = sys.argv[1]
test_path = sys.argv[2]

### read data
img = np.load(img_path)

img = img.astype('float32')/255
img = np.reshape(img, (len(img), -1))

X_train = img[:int(len(img)*train_ratio)]

X_valid = img[int(len(img)*train_ratio):]

print(X_train.shape)
print(X_valid.shape)

### build model
input_img = Input(shape=(784,))
encoded = Dense(256,activation='selu')(input_img)
encoded = Dense(128,activation='selu')(encoded)
encoded = Dense(64,activation='selu')(encoded)
encoded = Dense(32,activation='selu')(encoded)

decoded = Dense(64,activation='selu')(encoded)
decoded = Dense(128,activation='selu')(decoded)
decoded = Dense(256,activation='selu')(decoded)
decoded = Dense(784,activation='selu')(decoded)

### build encoder 

encoder = Model(input=input_img, output=encoded)

### build autoencoder 
adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

#autoencoder.fit(X_train, X_train, 
#	            epochs=epochs,
#	            batch_size=batch_size,
#	            shuffle=True,
#	            validation_data=(X_valid,X_valid))


#autoencoder.save('./autoencoder.h5')
#encoder.save('./encoder.h5')

autoencoder = keras.models.load_model('./autoencoder.h5')
encoder = keras.models.load_model('./encoder.h5')

encoded_imgs = encoder.predict(img)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)

db = KMeans(n_clusters=2).fit(encoded_imgs)

## get test data#
f = pd.read_csv(test_path)
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

## predict
print('Predicting...')
ans = open(sys.argv[3],'w')
ans.write('ID,Ans\n')

for idx, i1, i2 in zip(IDs, idx1, idx2):
	p1 = db.labels_[i1]
	p2 = db.labels_[i2]

	if p1==p2:
		pred = 1
	else:
		pred = 0	
	ans.write('{},{}\n'.format(idx,pred)) 
ans.close()

#X_embedded = TSNE(n_components=2).fit_transform(encoded_imgs)

#data_A = []
#data_B = []
#for i in range(len(db.labels_)):
#	if i == 1:
#		data_A.append(encoded_imgs[i])
#	else:
#		data_B.append(encoded_imgs[i])
#plt.scatter(data_A[:,0], data_A[:,1], c='b', label='dataset A', s= 0.2)
#plt.scatter(data_B[:,0], data_B[:,1], c='r', label='dataset B', s= 0.2)
#plt.legend()
#plt.savefig('prediction_tsne.jpg')
#
#plt.scatter(X_embedded[:5000, 0], X_embedded[:5000, 1], c='b', label='dataset A', s= 0.2)
#plt.scatter(X_embedded[5000:, 0], X_embedded[5000:, 1], c='r', label='dataset B', s= 0.2)
#plt.legend()
#plt.savefig('tsne.jpg')


