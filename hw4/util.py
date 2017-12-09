import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
import gensim

class DataManager:
    def __init__(self):
        self.data = {}
        self.test_data = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    X.append(line)
        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]
    
    def add_test_data(self,name,test_path):
        print('read testing data from %s...'%test_path)
        T = []
        with open(test_path,'r') as f:
            for line in f:
                line = line.strip('\n')
                filter_index = line.find(',',)
                T.append(line[filter_index+1:])
        T=T[1:]
        self.test_data[name] = [T]
     
    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size, 
                                   filters='', 
                                   lower=True, 
                                   char_level=False)
        #self.tokenizer = Tokenizer(num_words=vocab_size, lower=True, char_level=False)
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))


    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
        for k in self.test_data:
            print ('Converting %s to sequences'%k)
            temp = self.tokenizer.texts_to_sequences(self.test_data[k][0])
            self.test_data[k][0] = np.array(pad_sequences(temp, maxlen=maxlen))

    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to count'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
        
        for k in self.test_data:
            print ('Converting %s to count'%k)
            self.test_data[k][0] = self.tokenizer.texts_to_matrix(self.test_data[k][0],mode='count')


    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))
                
    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

    # get data by name
    def get_data(self,name):
        return self.data[name]
    
    def get_test_data(self, name):
        return self.test_data[name][0]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
