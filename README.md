# lstm-language-model

该库使用了lstm算法来训练一个语言模型，训练语料是经过筛选的wikipedia语料。

### 开发环境 
+ python 3.7.9
+ nltk==3.5
+ Keras==2.4.3
+ numpy==1.18.5
+ tensorflow-gpu==2.3.1

### 使用方法

+ 数据预处理
    + 使用的数据是经过预处理的Wikipedia数据，如何预处理的数据，请参加[wikiDataPreprocess](https://github.com/ChenTao98/wikiDataPreprocess)。得到预处理数据后，进行分句分词，筛选处理，选择最大token数量不超过30，最小不超过3的句子。运行命令如下：
    ```
    python data_preprocess.py -i .\data\wikipedia -o .\data\wikipediaCut.txt
    ```
    + 分句之后将句子token转为id，并进行清洗，清洗包含未知token的句子。使用的词表是[bert](https://github.com/google-research/bert)的英文词表，并添加了[UNK]token
    ```
    python datatoid.py
    python data_clean.py
    ```

+ 训练
    + 数据预处理之后，即可进行训练
    ```
    python lstm_model.py
    ```
    + 训练结束后，可使用ppl-cal.py计算句子ppl，注意更换ppl-cal.py中的weight文件