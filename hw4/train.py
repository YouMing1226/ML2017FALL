import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import gensim
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


from util import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--nb_epoch', default=10, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=36,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=256, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.01,type=float)
parser.add_argument('--threshold', default=0.3,type=float)

# output path for your prediction
parser.add_argument('--result_path', default= sys.argv[6])

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
args = parser.parse_args()

train_path = sys.argv[3]
test_path = sys.argv[5]
semi_path = sys.argv[4]

# build model
def simpleRNN(args,emb_mat):
    inputs = Input(shape=(args.max_length,))

    # Embedding layer
    embedding_inputs = Embedding(30674, 
                                 args.embedding_dim,
                                 weights=[emb_mat],
                                 trainable= False)(inputs)
    # RNN 
    return_sequence = True
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size,
                       return_sequences=return_sequences,  
                       recurrent_dropout=dropout_rate,
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size,
                        return_sequences=return_sequence,
                        recurrent_dropout=dropout_rate, 
                        dropout=dropout_rate)

    RNN_output = LSTM(args.hidden_size,
                      return_sequences=return_sequence,
                      recurrent_dropout=dropout_rate,
                      dropout=dropout_rate)(embedding_inputs)
    #RNN_output = RNN_cell(RNN_output)
    
    return_sequence = False
   
    RNN_output = LSTM(args.hidden_size,
                      return_sequences=return_sequence,
                      recurrent_dropout=dropout_rate,
                      dropout=dropout_rate)(RNN_output)
    
    #RNN_output = LSTM(args.hidden_size, 
    #                  return_sequences= True, 
    #                  dropout=dropout_rate)(embedding_inputs)
    #RNN_output = LSTM(args.hidden_size,
    #                  return_sequences=return_sequence,
    #                  dropout=dropout_rate)(RNN_output)
    
    # DNN layer
    outputs = Dense(args.hidden_size//2,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(args.hidden_size//4,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile(loss=args.loss_function, optimizer=adam, metrics=[ 'accuracy',])

    return model

def main():
   # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session(args.gpu_fraction))

    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)


#####read data#####
    dm = DataManager() 
    w2v_path = os.path.join(save_path,'word2vec')
    
    if args.action == 'train': 
        print ('Loading data...')
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
        dm.add_test_data('test_data',test_path)
    
        test_data = dm.get_test_data('test_data')  
        train_data = dm.get_data('train_data')
        semi_data = dm.get_data('semi_data')
        
        all_text = np.concatenate((train_data[0], semi_data[0], test_data),axis=0)
        print('Number of all_text:',all_text.shape[0])
        #print('Text sample:',all_text[0]) 
     
        print('Converting texts to words sequence...')
        text2word = []
   
        with_filter = 0 
        if with_filter: 
            for text in all_text:
                text2word.append(text_to_word_sequence(text,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" "))
        if not with_filter:
            for text in all_text:
                text2word.append(text_to_word_sequence(text,
                                 filters='',
                                 lower=True,
                                 split=" "))
   
        print('Word sequence sample:',text2word[0])

        if os.path.exists(w2v_path):
            print ('Loading w2v_model from %s' % w2v_path)
            word_vec = gensim.models.Word2Vec.load(w2v_path)
            print('Vocabulary size:',len(word_vec.wv.vocab))
        else:
            print ('Building word2vec model...')
            word_vec = gensim.models.Word2Vec(text2word, size=128, min_count=15)
            print('Vocabulary size:',len(word_vec.wv.vocab)) 
            if not os.path.isdir(save_path):  
                os.makedirs(save_path)
            if not os.path.exists(os.path.join(save_path,'word2vec')):
                word_vec.save((os.path.join(save_path,'word2vec')))
    
        print('Coverting train_data to vector...') 
        index_data = []
        i = 0
        for line in train_data[0]: 
            index_data.append([])
            for word in line.split():
                if word in word_vec.wv:
                    #print(word ,word_vec.wv.vocab[word].index)
                    index_data[i].append(word_vec.wv.vocab[word].index) 
            i+=1
    
         
        embedding_matrix = np.zeros((len(word_vec.wv.vocab),128))
   
        for i in range(len(word_vec.wv.vocab)):
            embedding_vector = word_vec.wv[word_vec.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        index_data = pad_sequences(index_data, args.max_length)
    else:
        if os.path.exists(w2v_path):
            print ('Loading w2v_model from %s' % w2v_path)
            word_vec = gensim.models.Word2Vec.load(w2v_path)
            print('Vocabulary size:',len(word_vec.wv.vocab))
            embedding_matrix = np.zeros((len(word_vec.wv.vocab),128))

            for i in range(len(word_vec.wv.vocab)):
                embedding_vector = word_vec.wv[word_vec.wv.index2word[i]]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector 
        else:
            print('Can not load w2v model, please training w2v model first!')


    #print ('get Tokenizer...')
    #if args.load_model is not None:
    #    # read exist tokenizer
    #    dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    #else:
    #    # create tokenizer on new data
    #    dm.tokenize(args.vocab_size)
    #                        
    #if not os.path.isdir(save_path):
    #    os.makedirs(save_path)
    #if not os.path.exists(os.path.join(save_path,'token.pk')):
    #    dm.save_tokenizer(os.path.join(save_path,'token.pk')) 
   # 
   # mat_train_data = dm.tokenizer.texts_to_matrix(train_data[0], mode='count')
   # mat_test_data = dm.tokenizer.texts_to_matrix(test_data, mode='count')

    # convert to sequences
    #dm.to_sequence(args.max_length)
  

 # initial model
    print ('initial model...')
    #model = bow_model(args,mat_train_data)
    model = simpleRNN(args, embedding_matrix)
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')

 # training
    if args.action == 'train':
        #(X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        X, X_val, Y, Y_val = train_test_split(index_data, train_data[1], test_size=0.33, random_state=42)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )
       
        print(history.history.keys())
        print('Val_acc:',history.history['val_acc'])
        print('Train_acc:',history.history['acc'])

 # testing
    elif args.action == 'test' :
        dm.add_test_data('test_data',test_path)
        test_data = dm.get_test_data('test_data')
        
        # Covert to vector
        index_test_data = []
        i = 0
        for line in test_data:
            index_test_data.append([])
            for word in line.split():
                if word in word_vec.wv:
                    #print(word ,word_vec.wv.vocab[word].index)
                    index_test_data[i].append(word_vec.wv.vocab[word].index)
            i += 1
        
        index_test_data = pad_sequences(index_test_data, args.max_length)
        
        csv_path = os.path.join(load_path,args.result_path)
        
        print("Predicting testing data...")
        Y_pred = model.predict(index_test_data)
        Y_pred = np.round(Y_pred)
        print('Saving result csv to',csv_path)
        with open(csv_path, 'w') as f:
            f.write('id,label\n')
            for i, v in  enumerate(Y_pred):
                f.write('%d,%d\n' % (i, v))

 # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' ) 
        # repeat 10 times
        for i in range(5):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=2048, verbose=True)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=256,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)


if __name__ == '__main__':
        main()


