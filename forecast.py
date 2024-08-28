import os

import jieba
from bidict import bidict

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_vocab(vocab_file):
    """
    从给定的词汇文件加载词汇表，并返回一个双向映射（bidict）对象。
    
    Args:
        vocab_file (str): 词汇文件路径。
    
    Returns:
        bidict: 词汇表，使用双向映射（bidict）对象表示。其中，键为词汇，值为词汇的索引。
    
    """
    vocab = bidict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

vocab = load_vocab("vocab/vocab.txt")
label_vocab = load_vocab("vocab/label_vocab.txt")

def text2index(text):
    """
    将文本转换为词汇索引列表
    
    Args:
        text (str): 待转换的文本
    
    Returns:
        List[int]: 词汇索引列表，其中未出现在词汇表中的词使用<UNK>代替
    
    """
    words = jieba.cut(str(text))
    indexes = []
    unk_index = vocab.get('<UNK>', None)
    if unk_index is None:
        unk_index = len(vocab)
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
        else:
            indexes.append(unk_index)
    return indexes

def index2label(index):
    """
    将整数型索引转换成对应的字符串型标签
    
    Args:
        index (int): 待转换的整数型索引
    
    Returns:
        str: 对应的字符串型标签
    
    """
    label_vocab_inv = label_vocab.inverse
    return label_vocab_inv[index]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, word_count, embedding_dim, num_layers, hidden_size, output_size, dropout_rate):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(word_count, embedding_dim)                                          # 词嵌入
        self.dropout_emb = nn.Dropout(dropout_rate)                                                       # 词嵌入dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True) # 双向LSTM
        self.dropout_lstm = nn.Dropout(dropout_rate)                                                      # LSTM dropout
        self.fc = nn.Linear(hidden_size * 2, output_size)                                                 # 全连接层

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout_emb(embedded)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout_lstm(lstm_out)
        lstm_last_step = lstm_out[:, -1, :]
        logits = self.fc(lstm_last_step)
        return F.log_softmax(logits, dim=1)

def prediction(text):
    tensor = torch.LongTensor(text2index(text)).unsqueeze(0).to(DEVICE)
    output = model(tensor)
    probs = F.softmax(output, dim=1)
    _, max_index = torch.max(probs, dim=1)
    return index2label(max_index.item()), probs[0, max_index.item()].item()

model = torch.load("model/lstm_model_early_stop.pth").to(DEVICE).eval()



test = ['我真的很喜欢你',
        '我很讨厌你',
        '你很好',
        '早年的警察情结立马涌现，悔得我啊~早知道伸腿绊他一下多好！',
        '想起了那间教室，想起了那条路，想起了那群人儿，勾起了我们人生中最青涩的记忆...', 
        '妈的感觉被蹭网了 刚刚看球赛好好的 现在连围脖 邮箱 什么也打不开 要命 哪个鸟 人',
        '但老狗一直坚信，人还是穿着衣服时漂亮，至少是穿着点什么的时候。',
        '鬼片真是吓死人了！！']

for text in test:
    print(f'{text} -> {prediction(text)}')