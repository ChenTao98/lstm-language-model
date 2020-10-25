import nltk
import nltk.tokenize as tk
import time,sys,os,getopt

def preprocess(input_file,out_file,max_len,min_len):
    with open(input_file,"r",encoding="utf-8") as input_fp:
        with open(out_file,"w",encoding="utf-8") as out_fp:
            for line in input_fp:
                sentences=tk.sent_tokenize(line.strip())
                for sent in sentences:
                    word=tk.word_tokenize(sent)
                    length=len(word)
                    if (length<max_len and length>=min_len):
                        out_fp.write(sent.lower()+"\n")

if __name__ == "__main__":
    opt,args=getopt.getopt(sys.argv[1:],"i:o:",["max_len=","min_len="])
    input_file=None
    out_file=None
    max_len=30
    min_len=3
    for o,a in opt:
        if(o=="-i"):input_file=a
        if(o=="-o"):out_file=a
        if(o=="--max_len"):max_len=int(a)
        if(o=="--min_len"):min_len=int(a)
    if(input_file==None or out_file==None):
        print("use argument -i as input file,-o as out file\n \
            --max_len as maximum number of tokens contained in a sentence  \n \
            --min_len as minimum number of tokens contained in a sentence")
        sys.exit()
    if((not os.path.exists(input_file)) or (not os.path.isfile(input_file))):
        print("use argument input_file not exists or is not a file")
        sys.exit()
    if(os.path.exists(out_file)):
        print("file {} has existed".format(out_file))
        sys.exit()
    start=time.time()
    preprocess(input_file,out_file,max_len,min_len)
    end=time.time()
    print("total time: %0.2fs" %(end-start))