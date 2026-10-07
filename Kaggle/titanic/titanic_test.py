# 训练 / 验证集划分版（8:2）——用来检查"模型是真学到规律，还是把训练数据背下来了"
#
# 和你 titanic_train.py 的结构完全一样，只有一处新增：
#   把 train.csv 的 891 行按 8:2 拆开
#     · 前 80%（约 713 行）当训练集，用来训练
#     · 后 20%（约 178 行）当验证集，用来"考试"
#
# 为什么要这么干：训练集准确率会虚高（模型可能只是记住了这 891 条），
# 只有在它没见过的验证集上评估，才知道真实水平；两个数字的差距就是"过拟合"的信号。

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = r'D:\Python_code\Kaggle_Titanic\titanic\train.csv'
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def load_dataframe(filepath):
    """读 csv + 文字列映射成数字 + 填缺失值（训练和验证都调用这一个函数，保证处理一致）"""
    df = pd.read_csv(filepath)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})            # 男→0，女→1
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # 三个港口→0/1/2
    df['Age'] = df['Age'].fillna(df['Age'].mean())                 # Age 缺 177 个，用均值补
    df['Embarked'] = df['Embarked'].fillna(0)                      # 缺 2 个，用最常见的 S(也就是 0) 补
    return df


class TitanicDataset(Dataset):
    """把 DataFrame 里的 7 个特征列变成 x，把 Survived 列变成 y"""
    def __init__(self, df, features):
        self.len = df.shape[0]                                   # 数据条数，__len__ 要用
        self.x_data = torch.from_numpy(df[features].values.astype(np.float32))
        self.y_data = torch.from_numpy(df[['Survived']].values.astype(np.float32))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ---------------- 1) 读数据 → 打乱 → 切 8:2 ----------------
df = load_dataframe(DATA_PATH)

np.random.seed(0)                                     # 固定种子：每次划分出的训练/验证集都是同一份
shuffled_index = np.random.permutation(df.index)      # 把行号打乱（不然前 891 行的顺序其实有规律）
split_point = int(len(df) * 0.8)                      # 80% 的位置
train_df = df.loc[shuffled_index[:split_point]].reset_index(drop=True)
val_df = df.loc[shuffled_index[split_point:]].reset_index(drop=True)
print('训练集 %d 条，验证集 %d 条' % (len(train_df), len(val_df)))

# 标准化：每列减均值、除标准差。注意均值和标准差只从"训练集"算，
# 验证集用的是训练集的同一组数字（将来预测 test.csv 时也要这样）。
# 这一步是关键：Fare 范围 0~512、Pclass 只有 1~3，尺度差一百多倍，不缩放模型几乎学不动。
mean = train_df[FEATURES].mean()
std = train_df[FEATURES].std()
train_df[FEATURES] = (train_df[FEATURES] - mean) / std
val_df[FEATURES] = (val_df[FEATURES] - mean) / std

train_dataset = TitanicDataset(train_df, FEATURES)
val_dataset = TitanicDataset(val_df, FEATURES)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)


# ---------------- 2) 模型（和你原来的完全一样：7 → 5 → 3 → 1） ----------------
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


def evaluate(dataset):
    """在给定数据集上算准确率：只做前向，不需要梯度"""
    with torch.no_grad():
        predicted = (model(dataset.x_data) > 0.5).float()   # 概率 > 0.5 判生还
        acc = (predicted == dataset.y_data).float().mean().item()
        rate = predicted.mean().item()                      # 预测生还的比例，用来自检
    return acc * 100, rate * 100


if __name__ == '__main__':
    for epoch in range(300):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            if epoch % 10 == 0 and i == 0:                  # 每 10 轮打印一次，避免刷屏
                print('epoch %3d   loss = %.4f' % (epoch, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_acc, train_rate = evaluate(train_dataset)
    val_acc, val_rate = evaluate(val_dataset)

    print()
    print('训练集准确率 = %.2f%%   (预测生还比例 %.1f%%，真实 38.4%%)' % (train_acc, train_rate))
    print('验证集准确率 = %.2f%%   (预测生还比例 %.1f%%，真实 %.1f%%)'
          % (val_acc, val_rate, val_df['Survived'].mean() * 100))
    print('两者差距     = %.2f 个百分点' % (train_acc - val_acc))