import numpy as np
import random 
import nltk.tokenize as tk
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku 
import csv
import time
with open('data\\vocab.txt','r',encoding='utf-8') as f:
	id2w=[line.strip() for line in f.readlines()]
	w2id={v:k for k,v in enumerate(id2w)}

max_len=30
data=[]
start=time.time()
with open('data\\wikipediaCut.txt','r',encoding='utf-8') as f:
	with open("data\\data_to_id.csv","w",newline="",encoding="utf-8") as fp:
		writer=csv.writer(fp)
		for line in f.readlines():
			words=tk.word_tokenize(line.strip())
			if len(words)>max_len:words=words[:max_len]
			sentence=[w2id.get(i,1) for i in words]
			writer.writerow(sentence)
now2=time.time()
print("store data cost:"+str(now2-start))