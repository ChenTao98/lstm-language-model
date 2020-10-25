import numpy as np
import random 
import nltk.tokenize as tk
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku
import time
import csv
with open('data\\vocab.txt','r',encoding='utf-8') as f:
	id2w=[line.strip() for line in f.readlines()]
	w2id={v:k for k,v in enumerate(id2w)}

context=3
max_len=30
data=[]
now = time.time()
with open('data\\data_to_id_train.csv','r',encoding='utf-8') as f:
	reader=csv.reader(f)
	for line in reader:
		sentence=np.array(line,dtype=int)
		temp_list=list()
		for i in range(0,len(sentence)):
			temp_list.append(sentence[:i+1])
		temp_list=pad_sequences(temp_list,maxlen=max_len,padding="pre")
		for i in temp_list:
			data.append(i)
print("data length:"+str(len(data)))
now2=time.time()
print("get data cost:"+str(now2-now))
def generator(batch_size, data=data):
	count=0
	data_length=len(data)
	while True:
		if(count+batch_size>=data_length):
			count=0
		tmp=data[count:count+batch_size]
		count+=batch_size
		print(count)
		result=np.array(tmp)
		predictors, label = result[:,:-1],result[:,-1]
		label = ku.to_categorical(label, num_classes=len(w2id))
		yield predictors,label