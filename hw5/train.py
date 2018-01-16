import time
import pandas as pd
import numpy as np
import os, sys
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, Dot, Add, Input, Merge, Embedding, Concatenate
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from sklearn.manifold import TSNE

# movieID::Title::Genres
movie_path = 'data/movies.csv'

# UserID::Gender::Age::Occupation::Zip-code
user_path = 'data/users.csv'

# TestDataID,UserID,MovieID
test_path = sys.argv[3]
print('Loading test data from {}'.format(test_path))

# TrainDataID,UserID,MovieID,Rating
train_path = './data/train.csv'

#def load_data(train_path, test_path):
print('Loading data...')

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, index_col=None)
    test_data = pd.read_csv(test_path, index_col=None)

    # test data
    user_test = test_data['UserID'].values
    movie_test = test_data['MovieID'].values

    # data
    user_data = train_data['UserID'].values
    movie_data = train_data['MovieID'].values
    y = train_data['Rating'].values
   
    #y_mean = np.mean(y)
    #y_std = np.std(y)
    #y = (y-y_mean)/y_std
   
    indices = np.arange(user_data.shape[0])
    np.random.shuffle(indices)
    user_data = user_data[indices]
    movie_data = movie_data[indices]
    y = y[indices]

    user_train, user_val, movie_train, movie_val, y_train, y_valid = train_test_split(user_data,
                                                                                      movie_data, 
                                                                                      y,
                                                                                      test_size =0.1,
                                                                                      random_state=42)

    print('Use train:',len(user_train))
    print('Movie train:',len(movie_train))    

    return (user_data, movie_data, user_train, user_val, movie_train, movie_val, y_train, y_valid, user_test, movie_test)

def mf_model(n_users, n_items, latent_dim, drop_rate):
    #print(n_users, n_items, latent_dim, drop_rate)
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])

    user_vec = Embedding(n_users, latent_dim, input_length=1)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(drop_rate)(user_vec)

    item_vec = Embedding(n_items, latent_dim, input_length=1)(item_input)
    item_vec = Flatten()(item_vec)
    item_vec = Dropout(drop_rate)(item_vec)

    user_bias = Embedding(n_users, 1, input_length=1)(user_input)
    user_bias = Flatten()(user_bias)
    user_bias = Dropout(drop_rate)(user_bias)

    item_bias = Embedding(n_items, 1, input_length=1)(item_input)
    item_bias = Flatten()(item_bias)
    item_bias = Dropout(drop_rate)(item_bias)

    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model(inputs=[user_input, item_input], outputs=r_hat)
    model.compile(loss='mse', optimizer='Adadelta')
    model.summary()
    
    return model

def nn_model(n_users, n_items, latent_dim, drop_rate):
    user_input = Input([1])
    item_input = Input([1])
    user_vec = Embedding(n_users, latent_dim, input_length=1)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(drop_rate)(user_vec)

    item_vec = Embedding(n_items, latent_dim, input_length=1)(item_input)
    item_vec = Flatten()(item_vec)
    item_vec = Dropout(drop_rate)(item_vec)

    merge_vec = Concatenate()([user_vec, item_vec])

    hidden = Dense(128, activation = 'relu')(merge_vec)
    #hidden = Dense(128, activation = 'relu')(hidden)
    hidden = Dense(64, activation = 'relu')(hidden)
    #hidden = Dense(32, activation = 'relu')(hidden)
    output = Dense(1)(hidden)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(loss='mse', optimizer='Adadelta')
    print(model.summary())

    return model

start_time = time.time()

user_data, movie_data, user_train, user_val, movie_train, movie_val, y_train, y_valid, user_test, movie_test = load_data(train_path, test_path)

n_users = max(user_data)
n_movies = max(movie_data)

latent_dim = int(sys.argv[2])
batch_size = 2048
epoch = 500
dropout_rate = 0.1

print('Train users:',n_users)
print('Train movies:',n_movies)
print('Latent dim:',latent_dim)

#model_dir = 'model'
#if not os.path.isdir(model_dir):
#    os.makedirs(model_dir)

model_path = 'model_'+str(latent_dim)+'.h5'

# training

if sys.argv[1] == 'train':
    checkpoint  = ModelCheckpoint(filepath=model_path, verbose=1, monitor='val_loss', save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')

    model = mf_model(n_users, n_movies, latent_dim, dropout_rate)
    #model = nn_model(n_users, n_movies, latent_dim, dropout_rate)
    print('Training model...')
    history = model.fit([user_train, movie_train], y_train, epochs=epoch, batch_size=batch_size,
                    validation_data=([user_val, movie_val], y_valid),
                    verbose=1, callbacks=[checkpoint, earlystopping])

    #train_acc = history.history['acc']
    #valid_acc = history.history['val_acc']
    #train_loss = history.history['loss']
    #valid_loss = history.history['val_loss']
    #print('Train accuracy:',train_acc)
    #print('Valid accuracy:',valid_acc)
    #print('Train loss:',train_loss)
    #print('Valid loss:',valid_loss)
    model.save(model_path)

# testing
if sys.argv[1] == 'test':
    print('Loading model from',model_path)
    
    test_data = pd.read_csv(test_path, index_col=None)
    user_test = test_data['UserID'].values
    movie_test = test_data['MovieID'].values

    #result_dir = 'answer'
    #if not os.path.isdir(result_dir):
    #    os.makedirs(result_dir)

    #result_file  = 'pred_'+str(latent_dim)+'.csv'
    #save_path = os.path.join(result_dir,result_file)
    save_path = sys.argv[4]
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    model = load_model(model_path)

    y_pred = model.predict([user_test,movie_test], verbose=1)
    #y_pred = y_pred * y_std + y_mean 
    #print(y_pred[0:5])
    
    print('\nSaving csv to',save_path)
    save_file = os.path.join(save_path,'prediction.csv')
    with open(save_file, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range (y_pred.shape[0]):
            if (y_pred[i][0] > 5):
                f.write(str(i+1) + ",5\n")
            elif (y_pred[i][0] < 1):
                f.write(str(i+1) + ",1\n")
            else:
                f.write(str(i+1) + "," + str(y_pred[i][0]) + "\n")

print(time.time() - start_time)


