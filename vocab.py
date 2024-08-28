import jieba
import pandas as pd

df = pd.read_csv('dataset/train.csv')

data = df['文本'].values

label = df['标签'].values

vocab_label = set(label)
vocab_label = sorted(list(vocab_label))
with open('vocab/label_vocab.txt', 'w', encoding='utf-8') as f:
    for word in vocab_label:
        f.write(word + '\n')

vocab = set()

for i in data:
    words = jieba.cut(str(i))
    for word in words:
        vocab.add(word)

vocab.add('<UNK>')
vocab.add('<PAD>')

vocab = sorted(list(vocab))

with open('vocab/vocab.txt', 'w', encoding='utf-8') as f:
    for word in vocab:
        f.write(word + '\n')