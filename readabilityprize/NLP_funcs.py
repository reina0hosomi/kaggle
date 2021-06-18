import pandas as pd
import numpy as np
from pickle import dump, load

#--------
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,\
Dropout,BatchNormalization,Input,Conv1D,MaxPool1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

#--------
class NLP_Embedding():
    def __init__(self,train_seri,train_y):
        self.train_seri = train_seri
        self.train_y = train_y
        self.train_texts_list = []
        self.tokenizer = Tokenizer()
        self.max_length = None
        self.vocab_size = None
        self.glove_embedding = {}
        self.test_size=0.3
        
    def loadGloves(self):
        glove_embedding_file =  "./glove/glove.6B.300d.txt"
        f = open(glove_embedding_file)
        for l in f :
            content = l.split()
            self.glove_embedding[content[0]] = np.asarray(content[1:])
            
    def __seriToList(self):
        for s in self.train_seri:
            self.train_texts_list.append(s)
        return self.train_texts_list
    
    def MakeTokenizer(self):
        self.__seriToList()
        self.tokenizer.fit_on_texts(self.train_texts_list)
        dump(self.tokenizer, open('tokenizer.pkl', 'wb'))
        return 
    
    def GetVocabSize(self):
        vocab_size = len(self.tokenizer.word_index) 
        print("Vocabulary Size of Texts:　", self.vocab_size)
        return
    
    def GetMaxLength(self):
        """
        ・訓練用の文の最大単語数を取得する。
        """
        self.max_length = max(len(d.split()) for d in self.train_texts_list)
        print("Max Length of Texts: ", self.max_length)
        return self.max_length
    
    def makeSequences(self):
        vocab_sizextr_seq = self.tokenizer.texts_to_sequences(self.train_texts_list)##単語を番号に変換
        word_index = self.tokenizer.word_index
        train_seq = pad_sequences(vocab_sizextr_seq,maxlen=self.max_length,padding="post",truncating="post")##行方向：文書index 列：単語
        return train_seq
    
    def createModel(self):
        self.loadGloves()
        ###embetting_matrix (縦：vocaburarysize よこ:次元数)
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index)+1,300))
        for word,i in word_index.items() :
            if self.glove_embedding.get(word) is not None:
                embedding_matrix[i,:] = self.glove_embedding.get(word)
        ###
        embedding_layer = Embedding(len(word_index)+1,300,weights=[embedding_matrix],input_length=\
                           self.max_length,trainable=False)

        model = Sequential([embedding_layer,Bidirectional(LSTM(64,return_sequences=True)),\
                   Bidirectional(LSTM(64,)),Dense(1)])
        model.compile(loss="mse",optimizer="adam",metrics="mse")
        
        return model
    
    def trainModel(self):
        self.MakeTokenizer()
        self.GetMaxLength()
        train_seq = self.makeSequences()
        model = self.createModel()
        checkpoint_lstm = './lstm/checkpoint'
        early_stopping = EarlyStopping(patience=10,min_delta=0.1,monitor="val_loss")
        check_point = ModelCheckpoint(checkpoint_lstm,monitor="val_loss",save_weights_only=True,\
                             save_best_only=True,mode="min")
        ###学習7割 検証3割
        xtr_seq,xts_seq,ytr,yts = train_test_split(train_seq,self.train_y,test_size=self.test_size)
        his=model.fit(xtr_seq,ytr,validation_data=(xts_seq,yts),epochs=100,batch_size=32,\
          callbacks=[check_point])
        return
    def loadPredictModel(self,checkpoint_lstm='./lstm/checkpoint'):
        model = self.createModel()
        model.load_weights(checkpoint_lstm)
        return model
    
    def predict(self,test_x = None,model = None,token_file='./tokenizer.pkl',max_length=205):
        ts_X = test_x
        tokenizer = load(open(token_file, "rb"))
        X = tokenizer.texts_to_sequences(ts_X)
        X = pad_sequences(X,maxlen=max_length,padding="post",truncating="post")
        prediction = model.predict(X)
        return prediction


