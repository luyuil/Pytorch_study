# 用 reviews_train.py 里那套模型，对 test.tsv 做预测，生成可以上传 Kaggle 的 submission.csv
#
# 和 reviews_train.py 的三点区别：
#   1) 不再划分验证集：用全部 156060 条数据训练（提交当然要用上全部数据）；
#      轮数取验证集上表现最好的那次（第 10 轮），所以固定训 10 轮。
#   2) 多了一个"预测"流程：读 test.tsv（没有标签列）→ 用训练集的词表编码 →
#      前向传播 → 取概率最大的类别。
#   3) 最后把 PhraseId 和预测结果拼成两列，写成 submission.csv。

import csv
import math
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# ---------------- 超参数（和 reviews_train.py 保持一致） ----------------
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 10          # 验证集上第 10 轮成绩最好，所以全量数据也训 10 轮
MAX_LEN = 60
MIN_FREQ = 2
MAX_VOCAB = 20000
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
LR = 1e-3
USE_GPU = True

device = torch.device('cuda:0') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
torch.manual_seed(0)
np.random.seed(0)


# ========= 1) 数据集：有标签的 train.tsv 和没标签的 test.tsv 都能读 =========
class ReviewsDataset(Dataset):
    WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, filename, vocab=None, has_label=True):
        self.keys = []          # PhraseId，写提交文件时要用
        self.phrases = []
        self.sentiments = []    # 测试集没有这一列，就保持空列表
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)                       # 跳过表头
            for row in reader:
                self.keys.append(int(row[0]))
                self.phrases.append(row[2])
                if has_label:
                    self.sentiments.append(int(row[3]))
        self.len = len(self.phrases)

        # 词表：训练集自己建；测试集必须复用训练集那一份
        self.vocab = self._build_vocab(self.phrases) if vocab is None else vocab
        ReviewsDataset.vocab = self.vocab

    @classmethod
    def _build_vocab(cls, phrases):
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
        words = self.WORD_RE.findall(phrase.lower())[:MAX_LEN]
        ids = [self.vocab.get(w, 0) for w in words]
        return ids if ids else [0]

    def __getitem__(self, index):
        if self.sentiments:
            return self.phrases[index], self.sentiments[index]
        return self.phrases[index], 0          # 测试集没有标签，占位用

    def __len__(self):
        return self.len


# ========= 2) 把 phrase 变成张量 =========
def phraselist(phrase):
    words = ReviewsDataset.WORD_RE.findall(phrase.lower())[:MAX_LEN]
    ids = [ReviewsDataset.vocab.get(w, 0) for w in words]
    if not ids:
        ids = [0]
    return ids, len(ids)


def create_tensor(tensor):
    return tensor.to(device)


def make_tensors(phrases, sentiments):
    """比训练版多返回一个 perm_idx：因为按长度排序打乱了顺序，预测完要还原回来"""
    sequences_and_lengths = [phraselist(phrase) for phrase in phrases]
    phrase_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    sentiments = sentiments.long()

    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    sentiments = sentiments[perm_idx]

    return create_tensor(seq_tensor), seq_lengths, create_tensor(sentiments), perm_idx


# ========= 3) 模型（和 reviews_train.py 一致） =========
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.emb_dropout = torch.nn.Dropout(DROPOUT)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional, dropout=DROPOUT)
        self.fc = torch.nn.Linear(hidden_size * self.n_direction, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_size)
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


# ========= 4) 训练 / 预测 =========
def trainModel(epoch):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0
    for i, (phrases, sentiments) in enumerate(trainloader, 1):
        inputs, seq_lengths, target, _ = make_tensors(phrases, sentiments)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5)
        optimizer.step()

        total_loss += loss.item() * len(target)
        correct += (output.max(dim=1)[1] == target).sum().item()
        total += len(target)
        if i % 100 == 0:
            print(f'  [{time_since(start)}] epoch {epoch}  {total}/{len(trainset)}  loss={total_loss / total:.4f}')
    return total_loss / total, 100 * correct / total


def predict(loader, n):
    """对没有标签的数据做预测，返回长度 n 的类别数组"""
    classifier.eval()
    preds = np.zeros(n, dtype=np.int64)
    done = 0
    with torch.no_grad():
        for phrases, _ in loader:
            # 测试集没有标签，用一个全 0 的 LongTensor 占位（make_tensors 要求 tensor）
            dummy = torch.LongTensor([0] * len(phrases))
            inputs, seq_lengths, target, perm_idx = make_tensors(phrases, dummy)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1)[1].cpu()
            back = torch.empty_like(pred)
            back[perm_idx] = pred          # 还原成"原始顺序"
            preds[done:done + len(pred)] = back.numpy()
            done += len(pred)
    return preds


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ========= 5) 主流程 =========
if __name__ == '__main__':
    start = time.time()
    trainset = ReviewsDataset('train.tsv')                                  # 全量训练 + 建词表
    testset = ReviewsDataset('test.tsv', vocab=trainset.vocab, has_label=False)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    classifier = RNNClassifier(len(trainset.vocab), HIDDEN_SIZE, 5, N_LAYER)
    classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print('词表 %d 个词；训练 %d 条，测试 %d 条'
          % (len(trainset.vocab), len(trainset), len(testset)))

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = trainModel(epoch)
        print('epoch %2d  train_loss=%.4f  train_acc=%.2f%%   [%s]'
              % (epoch, tr_loss, tr_acc, time_since(start)), flush=True)

    preds = predict(testloader, len(testset))
    submission = pd.DataFrame({'PhraseId': testset.keys, 'Sentiment': preds})
    submission.to_csv('submission.csv', index=False)

    counts = np.bincount(preds, minlength=5)
    print()
    print('预测各类别数量:', counts.tolist())
    print('预测各类别占比: %s' % [round(x, 1) for x in counts / len(preds) * 100])
    print('训练集真实占比: [4.5, 17.5, 51.0, 21.1, 5.9]')
    print('已保存 submission.csv，共 %d 行' % len(submission))
    print(submission.head().to_string(index=False))
