# Sentiment-analysis
nlp情感分类

数据集采用NLpcc 2014, 不放入仓库，自行下载，[链接](https://github.com/DinghaoXi/chinese-sentiment-datasets)

采用Dropout正则，基于准确率的早停法，防止过拟合，已经有训练好的模型，可以直接预测，需要自己解压，采用7z极限压缩


|    文件     |    作用    |
| :---------: | :--------: |
|  merge.py   | 合并数据集 |
| forecast.py |    预测    |
|  train.py   |    训练    |
|  vocab.py   |  生成词表  |

|  文件夹  |   作用   |
| :------: | :------: |
| datasets |  数据集  |
|   log    | 日志文件 |
|  model   | 模型文件 |
|  vocab   | 词表文件 |

## 文件结构如下

```
./
│  .gitignore
│  forecast.py
│  LICENSE
│  merge.py
│  README.md
│  train.py
│  vocab.py
│
├─dataset
│  │  train.csv
│  │  train_noNone.csv
│  │
│  ├─.ipynb_checkpoints
│  │      train-checkpoint.csv
│  │
│  ├─Nlpcc2013
│  │      Nlpcc2013Train.tsv
│  │      Nlpcc2013Train_NoNone.tsv
│  │      WholeSentence_Nlpcc2013Train.tsv
│  │      WholeSentence_Nlpcc2013Train_NoNone.tsv
│  │
│  └─Nlpcc2014
│          Nlpcc2014Train.tsv
│          Nlpcc2014Train_NoNone.tsv
│          WholeSentence_Nlpcc2014Train.tsv
│          WholeSentence_Nlpcc2014Train_NoNone.tsv
│
├─log
├─model
└─vocab
        label_vocab.txt
        vocab.txt
```