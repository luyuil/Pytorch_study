# 用 titanic_test.py 里的同一套预处理和模型，训练完对 test.csv 做预测，
# 生成可以直接上传 Kaggle 的提交文件 submission.csv
#
# 和你 titanic_test.py 的三点差别：
#   1) 这次不划分验证集了，直接用全部 891 条有标签的数据训练（最终提交当然要用上全部数据）
#   2) 多了一步：读 test.csv（它没有 Survived 列），套用同一套预处理和同一组均值/标准差
#   3) 最后把预测结果写成 PassengerId,Survived 两列，存成 submission.csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = r'D:\Python_code\Kaggle_Titanic\titanic\train.csv'
TEST_PATH = r'D:\Python_code\Kaggle_Titanic\titanic\test.csv'
OUT_PATH = r'D:\Python_code\Kaggle_Titanic\submission.csv'
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

torch.manual_seed(0)     # 固定随机种子，结果可复现
np.random.seed(0)


def load_dataframe(filepath, age_fill=None, fare_fill=None):
    """读 csv + 文字列映射成数字 + 填缺失值
    age_fill / fare_fill 是"填缺失用的数字"：训练集不传，就用它自己的均值；
    预测 test.csv 时要把训练集的均值传进来，保证两边处理方式一致。"""
    df = pd.read_csv(filepath)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})            # 男→0，女→1
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # 三个港口→0/1/2
    if age_fill is None:
        age_fill = df['Age'].mean()
    if fare_fill is None:
        fare_fill = df['Fare'].mean()
    df['Age'] = df['Age'].fillna(age_fill)
    df['Fare'] = df['Fare'].fillna(fare_fill)   # test.csv 里有 1 个缺失的 Fare，不填就会变成 NaN
    df['Embarked'] = df['Embarked'].fillna(0)
    return df


class TitanicDataset(Dataset):
    def __init__(self, df, features):
        self.len = df.shape[0]
        self.x_data = torch.from_numpy(df[features].values.astype(np.float32))
        if 'Survived' in df.columns:
            self.y_data = torch.from_numpy(df[['Survived']].values.astype(np.float32))
        else:
            self.y_data = None          # test.csv 没有标签，用不到

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    """和 titanic_test.py 里完全一样：7 → 5 → 3 → 1，中间 ReLU、最后 sigmoid"""
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


# ---------------- 1) 读训练数据（全部 891 条） + 标准化 ----------------
train_df = load_dataframe(TRAIN_PATH)
age_mean = train_df['Age'].mean()      # 这两个均值要留给 test.csv 用
fare_mean = train_df['Fare'].mean()

mean = train_df[FEATURES].mean()       # 标准化用的均值和标准差，也只从训练集算
std = train_df[FEATURES].std()
train_df[FEATURES] = (train_df[FEATURES] - mean) / std

train_dataset = TitanicDataset(train_df, FEATURES)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
print('训练集 %d 条' % train_dataset.len)

# ---------------- 2) 训练（参数和 titanic_test.py 一致） ----------------
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(300):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        loss = criterion(model(inputs), labels)
        if epoch % 50 == 0 and i == 0:
            print('epoch %3d   loss = %.4f' % (epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---------------- 3) 读 test.csv，套用同一套预处理 ----------------
test_raw = pd.read_csv(TEST_PATH)
test_df = load_dataframe(TEST_PATH, age_fill=age_mean, fare_fill=fare_mean)
test_df[FEATURES] = (test_df[FEATURES] - mean) / std     # 用的是训练集的均值和标准差
print('test %d 条，特征里剩余的缺失值 %d 个' % (test_df.shape[0], test_df[FEATURES].isnull().sum().sum()))

# ---------------- 4) 预测 ----------------
with torch.no_grad():
    x_test = torch.from_numpy(test_df[FEATURES].values.astype(np.float32))
    prob = model(x_test)                          # 每位乘客的生还概率
    pred = (prob > 0.5).int().numpy().flatten()   # 概率 > 0.5 记 1，否则记 0

    train_acc = ((model(train_dataset.x_data) > 0.5).float() == train_dataset.y_data).float().mean().item()

# ---------------- 5) 写成 Kaggle 要求的提交格式 ----------------
submission = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': pred})
submission.to_csv(OUT_PATH, index=False)

print()
print('训练集准确率 = %.2f%%（这是训练集分数，会偏高，别当真实水平）' % (train_acc * 100))
print('预测生还 %d / %d = %.1f%%（训练集真实生还率 38.4%%）' % (pred.sum(), len(pred), pred.mean() * 100))
print('已保存: %s' % OUT_PATH)
print(submission.head().to_string(index=False))