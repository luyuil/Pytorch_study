import csv
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 15   # 【过拟合修复④】原来100轮：跑到50轮时训练loss已背到0.0005、验证准确率阴跌到53%——15轮配合早停足够
N_CHARS = 128
USE_GPU = True

#=========数据集读取=============
class ReviewsDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'train.tsv'
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            rows = list(reader)
            split = int(len(rows) * 0.8)
            rows = rows[:split] if is_train_set else rows[split:]
        self.phrases = [row[2] for row in rows]
        self.sentiments = [int(row[3]) for row in rows]
        self.len = len(self.phrases)

    def __getitem__(self, index):
        return self.phrases[index], self.sentiments[index]

    def __len__(self):
        return self.len

#======把 phrase 转化成 张量===========
def phraselist(phrase):
    # 【过拟合修复①】统一小写：'T'和't'原本是两个不同编号，模型把大小写差异当特征死记；
    # 截断120：phrase最长283字符但平均只有40，超长尾巴基本不损失信息，还把GRU步数砍半
    arr = [ord(c) for c in phrase.lower()][:120]
    return arr, len(arr)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

def make_tensors(phrases, sentiments):
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

    return create_tensor(seq_tensor), \
           seq_lengths, \
           create_tensor(sentiments)

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
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional, dropout=0.3)
        self.fc = torch.nn.Linear(hidden_size * self.n_direction, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_direction == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

# ---------------- 训练 / 测试 ----------------
def trainModel():
    total_loss = 0
    for i, (phrases, sentiments) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(phrases, sentiments)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        # 【过拟合修复③】梯度裁剪：把全部参数的梯度总范数限制在5以内，
        # 防止个别batch算出超大梯度把参数带飞，训练更稳
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss

def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (phrases, sentiments) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(phrases, sentiments)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    percent = '%.2f' % (100 * correct / total)
    print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total

# ---------------- 主循环 ----------------
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    trainset = ReviewsDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = ReviewsDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    N_CLASS = 5
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASS, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    # 【过拟合修复⑤】早停+保存最优：验证准确率连续3个epoch不创新高就提前收工，
    # 留下的是"验证集上表现最好的那一刻"的模型，而不是过拟合之后的模型
    best_acc = 0
    no_improve = 0
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)
        if acc > best_acc:
            best_acc, no_improve = acc, 0
            torch.save(classifier.state_dict(), 'best_model.pt')   # 把当前最优模型的参数存盘
        else:
            no_improve += 1
            if no_improve >= 3:
                print(f'Early stop at epoch {epoch}, best acc = {best_acc:.4f}')
                break

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
