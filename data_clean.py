import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import time
max_len=30
start=time.time()
with open('data\\data_to_id.csv','r',encoding='utf-8') as f:
    with open("data\\data_to_id_train.csv","w",newline="",encoding="utf-8") as clean_f:
        reader=csv.reader(f)
        writer=csv.writer(clean_f)
        for line in reader:
            sentence=np.array(line,dtype=int)
            if(1 in sentence):
                continue
            writer.writerow(sentence)

end=time.time()
print("total time="+str(end-start))