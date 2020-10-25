from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,load_model
import keras.utils as ku 
import numpy as np 
import time
import nltk.tokenize as tk

with open('data\\vocab.txt','r',encoding='utf-8') as f:
	id2w=[line.strip() for line in f.readlines()]
	w2id={v:k for k,v in enumerate(id2w)}

max_len=30
total_words=len(w2id)
model=load_model("weights.hdf5")

def seq2id(seq):
    words=tk.word_tokenize(seq)
    if len(words)>max_len:print("seq is too long")
    sentence=np.array([w2id.get(i,1) for i in words])
    temp_list=list()
    for i in range(0,len(sentence)):
        temp_list.append(sentence[:i+1])
    temp_list=pad_sequences(temp_list,maxlen=max_len,padding="pre")
    return np.array(temp_list),len(words)

def ppl_cal(seq):
    seqid,length=seq2id(seq.strip().lower())
    predictors, label = seqid[:,:-1],seqid[:,-1]
    p_pred=model.predict(predictors)
    ppl=0
    # for prob in p_pred:
    #     print(np.sum(prob))
    for i,prob in enumerate(p_pred):
        # print(label[i])
        # print(prob[label[i]])
        ppl+=np.log(prob[label[i]])
    return(-ppl/length)

# print("i love you:"+str(ppl_cal("I love you")))
# print("i although you:"+str(ppl_cal("I although you")))
# print("system though night:"+str(ppl_cal("system though night")))
# print("The weather is so nice today:"+str(ppl_cal("The weather is so nice today")))
# print("The weather is asked nice today:"+str(ppl_cal("The weather is asked nice today")))
# print("population local february although village asked:"+str(ppl_cal("population local february although village asked")))