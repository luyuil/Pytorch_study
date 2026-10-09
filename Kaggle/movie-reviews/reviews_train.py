import csv
import time
import math
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 20   # 【过拟合修复④】原来100轮：跑到50轮时训练loss已背到0.0005、验证准确率阴跌到53%——15轮配合早停足够
MAX_LEN = 60
MIN_FREQ = 2
MAX_VOCAB = 20000
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4      # L2 正则强度；0.1 太大了，会把权重压得过狠（实测 1e-4 合适）
LABEL_SMOOTHING = 0.1    # 原来漏了这个常量，下面 CrossEntropyLoss 用到了它
LR = 1e-3
PATIENCE = 4
USE_GPU = True

device = torch.device('cuda:0') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
torch.manual_seed(0)
np.random.seed(0)

#=========数据集读取=============
class ReviewsDataset(Dataset):

    WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, is_train_set=True, vocab=None):
        rows = self._read_file('train.tsv', is_train_set)
        self.phrases = [row[2] for row in rows]
        self.sentiments = [int(row[3]) for row in rows]
        self.len = len(self.phrases)

        # 词表：训练集自己建；验证集/测试集必须复用训练集那一份
        self.vocab = self._build_vocab(self.phrases) if vocab is None else vocab
        ReviewsDataset.vocab = self.vocab      # 挂到类上，好让外面的 phraselist() 也能用到

    @staticmethod
    def _read_file(filename, is_train_set):
        """按行顺序 8:2 切：同一句话的短语是连着的，不会跨集合，避免信息泄漏"""
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)                       # 跳过表头
            rows = list(reader)
        split = int(len(rows) * 0.8)
        return rows[:split] if is_train_set else rows[split:]

    @classmethod
    def _build_vocab(cls, phrases):
        """统计词频建词表，0 号留给 <unk>（未知词），也当补位符号用"""
        counter = Counter()
        for phrase in phrases:
            counter.update(set(cls.WORD_RE.findall(phrase.lower())))
        vocab = {'<unk>': 0}
        for word, cnt in counter.most_common(MAX_VOCAB):
            if cnt < MIN_FREQ:
                break
            vocab[word] = len(vocab)
        return vocab

    def to_ids(self, phrase):
        """把一句话编码成词编号数组（方法版，方便外部按需调用）"""
        words = self.WORD_RE.findall(phrase.lower())[:MAX_LEN]
        ids = [self.vocab.get(w, 0) for w in words]
        return ids if ids else [0]


    def __getitem__(self, index):
        return self.phrases[index], self.sentiments[index]

    def __len__(self):
        return self.len

#======把 phrase 转化成 张量===========
def phraselist(phrase):
    """原来是把每个字符转成 ord(c)；现在按词编号，用的是类上挂的那份词表"""
    words = ReviewsDataset.WORD_RE.findall(phrase.lower())[:MAX_LEN]
    ids = [ReviewsDataset.vocab.get(w, 0) for w in words]
    if not ids:
        ids = [0]
    return ids, len(ids)

def create_tensor(tensor):
    return tensor.to(device)

def make_tensors(phrases, sentiments):
    sequences_and_lengths = [phraselist(phrase) for phrase in phrases]
    phrase_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    sentiments = sentiments.long()

    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)   # pack 要求按长度降序
    seq_tensor = seq_tensor[perm_idx]
    sentiments = sentiments[perm_idx]

    return create_tensor(seq_tensor), seq_lengths, create_tensor(sentiments)

#===========模型==========
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # 【过拟合修复②】层间dropout=0.3：训练时随机丢弃30%的层间连接，
        # 逼模型不依赖某条固定通路，就没法逐条硬背训练集（层数>1时才生效，我们是2层）
        self.emb_dropout = torch.nn.Dropout(DROPOUT)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional, dropout=DROPOUT)
        self.fc = torch.nn.Linear(hidden_size * self.n_direction, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        hidden = self._init_hidden(input.size(1))
        embedding = self.emb_dropout(self.embedding(input))
        gru_input = pack_padded_sequence(embedding, seq_lengths)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_direction == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        return self.fc(hidden_cat)

# ---------------- 训练 / 测试 ----------------
def trainModel():
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0
    for i, (phrases, sentiments) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(phrases, sentiments)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item() * len(target)
        correct += (output.max(dim=1)[1] == target).sum().item()
        total += len(target)
        if i % 50 == 0:
            print(f'  [{time_since(start)}] epoch {epoch}  {total}/{len(trainset)}  loss={total_loss / total:.4f}')
    return total_loss / total, 100 * correct / total

def testModel(loader):
    classifier.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for phrases, sentiments in loader:
            inputs, seq_lengths, target = make_tensors(phrases, sentiments)
            output = classifier(inputs, seq_lengths)
            total_loss += criterion(output, target).item() * len(target)
            correct += (output.max(dim=1)[1] == target).sum().item()
            total += len(target)
    return total_loss / total, 100 * correct / total

# ---------------- 主循环 ----------------
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    # 读文件、建词表、编码现在都收进类里了，所以这里改成用类的接口：
    trainset = ReviewsDataset(is_train_set=True)                        # 训练集：自己建词表
    valset = ReviewsDataset(is_train_set=False, vocab=trainset.vocab)   # 验证集：复用训练集的词表
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    N_CLASS = 5
    classifier = RNNClassifier(len(trainset.vocab), HIDDEN_SIZE, N_CLASS, N_LAYER)
    classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=1)

    start = time.time()
    print('词表 %d 个词，训练 %d 条，验证 %d 条'
          % (len(trainset.vocab), len(trainset), len(valset)))
    print('epoch  train_loss  train_acc   val_loss   val_acc      lr')

    history = []
    best_acc, best_epoch, no_improve = 0.0, 0, 0
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = trainModel()
        va_loss, va_acc = testModel(valloader)
        history.append((tr_loss, tr_acc, va_loss, va_acc))
        lr_now = optimizer.param_groups[0]['lr']
        print('%5d  %9.4f  %8.2f%%  %9.4f  %7.2f%%   %.2e'
              % (epoch, tr_loss, tr_acc, va_loss, va_acc, lr_now), flush=True)

        scheduler.step(va_acc)
        if va_acc > best_acc:
            best_acc, best_epoch, no_improve = va_acc, epoch, 0
            torch.save(classifier.state_dict(), 'best_model.pt')     # 存"验证集最好"的那份
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print('早停：验证集自第 %d 轮起不再提升（最好成绩 %.2f%%）' % (best_epoch, best_acc))
                break

    # 【关键修复】把验证集上最好那一刻的权重加载回来，再评估/预测
    classifier.load_state_dict(torch.load('best_model.pt'))
    _, final_acc = testModel(valloader)
    print('验证集最佳准确率 = %.2f%%（第 %d 轮，已加载该权重）' % (best_acc, best_epoch))

    h = np.array(history)
    ep = np.arange(1, len(h) + 1)
    plt.figure()
    plt.plot(ep, h[:, 1], label='train acc')
    plt.plot(ep, h[:, 3], label='val acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()
    plt.figure()
    plt.plot(ep, h[:, 0], label='train loss')
    plt.plot(ep, h[:, 2], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid()
    plt.show()
