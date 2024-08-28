import os
import datetime

import jieba
import pandas as pd
import matplotlib.pyplot as plt
from bidict import bidict
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader


now = datetime.datetime.now()
year, month, day = now.year, now.month, now.day
count = 1
while True:
    log_file = f'log/{year}-{month}-{day}-{count}.log'
    if os.path.exists(log_file):
        count += 1
    else:
        break

f = open(log_file, 'w', encoding='utf-8')

df = pd.read_csv("dataset/train.csv")

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

def label2index(text):
    """
    将标签文本转换为标签索引。
    
    Args:
        text (str): 标签文本。
    
    Returns:
        int: 标签索引。如果标签文本在词汇表中，则返回对应的索引值；如果不在词汇表中，则返回词汇表长度。
    
    """
    if text in label_vocab:
        return label_vocab[text]
    else:
        return len(label_vocab)

df['data'] = df['文本'].apply(text2index)
df['label'] = df['标签'].apply(label2index)

data_numpy = df['data'].to_numpy()
labels_numpy = df['label'].to_numpy()

### 超参数  # TODO
BATCH_SIZE = 128                                                      # 批量大小
OUTPUT = len(label_vocab)                                             # 标签数量
WORD_COUNT = len(vocab)                                               # 词汇数量
EMBEDDING_DIM = 300                                                   # 词嵌入维度
HIDDEN_SIZE = 256                                                     # LSTM隐藏层维度
NUM_LAYERS = 5                                                        # LSTM层数
DROPOUT_RATE = 0.3                                                    # dropout比率
LR = 1e-3                                                             # 学习率
PATIENCE = 10                                                         # 早停轮数
MIN_DELTA = 0.0001                                                    # 最小变化
EPOCH = 700                                                           # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
SAVE_INTERVAL = 100                                                   # 保存模型间隔

f.write('=='*20 + 'Hyperparameters:' + '=='*20 + '\n')
f.write(f"BATCH SIZE: {BATCH_SIZE}\n")
f.write(f"OUTPUT: {OUTPUT}\n")
f.write(f"WORD COUNT: {WORD_COUNT}\n")
f.write(f"EMBEDDING DIM: {EMBEDDING_DIM}\n")
f.write(f"HIDDEN SIZE: {HIDDEN_SIZE}\n")
f.write(f"NUM LAYERS: {NUM_LAYERS}\n")
f.write(f"DROPOUT RATE: {DROPOUT_RATE:.2%}\n")
f.write(f"LR: {LR}\n")
f.write(f"PATIENCE: {PATIENCE}\n")
f.write(f"MIN DELTA: {MIN_DELTA}\n")
f.write(f"EPOCH: {EPOCH}\n")
f.write(f"DEVICE: {str(DEVICE)}\n")
f.write(f'SAVE INTERVAL: {SAVE_INTERVAL}\n')

def pad_and_tensorize(batch):  
    """
    对输入数据进行填充和tensor化
    
    Args:
        batch (list[tuple[list[int], int]]): 一个batch的数据，包含多个(文本序列, 标签)对，其中文本序列是int类型的列表，标签是int类型
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 返回填充后的文本序列tensor和标签tensor
    
    """
    texts, labels = zip(*batch)
    padded_texts = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(text, dtype=torch.long) for text in texts],
        batch_first=True,
        padding_value=vocab['<PAD>']
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

length = len(data_numpy)
label_length = len(labels_numpy)

train_data, test_data = data_numpy[:int(length * 0.9)], data_numpy[int(length * 0.9):]
train_label, test_label = labels_numpy[:int(length * 0.9)], labels_numpy[int(length * 0.9):]

train_dataset = MyDataset(train_data, train_label)
test_dataset = MyDataset(test_data, test_label)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_and_tensorize)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_and_tensorize)

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

model = Net(
    word_count=WORD_COUNT,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT,
    dropout_rate=DROPOUT_RATE
    ).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler() # 定义GradScaler对象
criterion = nn.NLLLoss()

f.write('=='*20 + 'Model structure' + '=='*20 + '\n')
f.write(f"MODEL:\n{repr(model)}\n")
f.write('=='*20 + 'Log' + '=='*20 + '\n')

class EarlyStopping():  
    def __init__(self, patience=5, min_delta=0.001):  
        """
        初始化EarlyStopping类。
        
        Args:
            patience (int): 容忍多少次验证集性能不提升后提前停止训练，默认为5。
            min_delta (float): 验证集性能的最小提升幅度，若小于此值则视为没有提升，默认为0.001。
        
        Returns:
            None
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = None
        self.early_stop = False

    def __call__(self, val_accuracy):  
        if self.best_accuracy is None:
            self.best_accuracy = val_accuracy
        elif val_accuracy - self.best_accuracy > self.min_delta:
            self.best_accuracy = val_accuracy
            # 如果验证准确性提高，则重置计数器
            self.counter = 0
        else:
            self.counter += 1
            tqdm.write(f"[{datetime.datetime.now():%H:%M:%S}] [WARN]: Early stopping counter {self.counter} of {self.patience}")
            f.write(f"[{datetime.datetime.now():%H:%M:%S}] [WARN]: Early stopping counter {self.counter} of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

os.system('cls' if os.name == 'nt' else 'clear')

train_loss_lst = []
test_loss_lst = []
acc_lst = []
entropy_lst = []
last_entropy = 0.
early_stop = False
try:
    for epoch in trange(EPOCH, desc='Epoch'):
        model.train()
        train_loss = 0.

        for batch in tqdm(train_loader, desc='Train', leave=False):
            texts, labels = batch
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(): # 自动混合精度训练
                log_probs = model(texts)
                loss = criterion(log_probs, labels)

            scaler.scale(loss).backward() # 对loss进行缩放，并反向传播
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_lst.append(train_loss)

        model.eval()
        test_loss = 0.
        correct = 0
        entropy = 0.

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Test', leave=False):
                texts, labels = batch
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                log_probs = model(texts)
                loss = criterion(log_probs, labels)
                _, preds = log_probs.max(dim=1)
                correct += (preds == labels).sum().item()
                test_loss += loss.item()

                entropy += -torch.mean(
                    torch.sum(torch.exp(log_probs) * log_probs, dim=1)  # 计算熵
                )

        test_loss /= len(test_loader)
        test_loss_lst.append(test_loss)

        acc = correct / len(test_dataset)
        acc_lst.append(acc)

        if early_stopping(acc):
            tqdm.write(f'[{datetime.datetime.now():%H:%M:%S}] [FATAL]: Early stopping, Saving model as [model/lstm_model_early_stop.pth]')
            f.write(f'[{datetime.datetime.now():%H:%M:%S}] [FATAL]: Early stopping, Saving model as [model/lstm_model_early_stop.pth]\n')
            early_stop = True
            torch.save(model, 'model/lstm_model_early_stop.pth')
            break

        entropy /= len(test_dataset)
        entropy_lst.append(entropy.cpu().item())
        entropy = round(entropy.cpu().item(), 4)
        entropy_change = entropy - last_entropy
        change = '+' if entropy_change > 0 else ''
        entropy_change = change + str(entropy_change)

        text = f'[{datetime.datetime.now():%H:%M:%S}] [INFO]: Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
        text += f'Accuracy: {acc:.2%}, Entropy: {entropy} ({entropy_change})'
        tqdm.write(text)
        f.write(text + '\n')
        
        last_entropy = entropy
        if (epoch + 1) % SAVE_INTERVAL == 0: # 每隔一定epoch保存一次模型
            tqdm.write(f'[{datetime.datetime.now():%H:%M:%S}] [INFO]: Saving model as [model/model_{epoch + 1}.pth]')
            f.write(f'[{datetime.datetime.now():%H:%M:%S}] [INFO]: Saving model as [model/model_{epoch + 1}.pth]\n')
            torch.save(model, f'model/model_{epoch + 1}.pth')
except Exception as e:
    print(f'[{datetime.datetime.now():%H:%M:%S}] [FATAL]: Error caught: {e}, STOP!!!!!  Saving model as [model/lstm_model_Error_Caught.pth]')
    f.write(f'[{datetime.datetime.now():%H:%M:%S}] [FATAL]: Error caught: {e}, STOP!!!!!  Saving model as [model/lstm_model_Error_Caught.pth]\n')
    torch.save(model, 'model/lstm_model_Error_Caught.pth')

if not early_stop:
    print(f'[{datetime.datetime.now():%H:%M:%S}] [INFO]: Training completed without early stopping. Saving model as [model/lstm_model.pth]')
    f.write(f'[{datetime.datetime.now():%H:%M:%S}] [INFO]: Training completed without early stopping. Saving model as [model/lstm_model.pth]\n')
    torch.save(model, 'model/lstm_model.pth')

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title('Train and Test Loss')
plt.plot(train_loss_lst, label='Train Loss')
plt.plot(test_loss_lst, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Accuracy')
plt.plot(acc_lst, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.title('Entropy')
plt.plot(entropy_lst, label='Entropy')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.legend()

plt.show()