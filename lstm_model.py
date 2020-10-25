from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku 
import numpy as np 
import time

from dataload import w2id,generator,data
max_len=30
total_words=len(w2id)
batch_size=512
steps_per_epoch=len(data)//batch_size
print("step per epoch:"+str(steps_per_epoch))
print("word list size:"+str(total_words))
def create_model():
	model = Sequential()
	model.add(Embedding(total_words, 100, input_length=max_len-1))
	model.add(LSTM(150, return_sequences = True))
	model.add(LSTM(100))
	model.add(Dense(total_words, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print (model.summary())
	return model
start=time.time()
gen=generator(batch_size)
save_weights=ModelCheckpoint("weights-{epoch:02d}-{loss:.4f}-{accuracy:.4f}.hdf5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=False)
model=create_model()
model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=10, callbacks=[save_weights])
print("model train cost:"+str(time.time()-start))
