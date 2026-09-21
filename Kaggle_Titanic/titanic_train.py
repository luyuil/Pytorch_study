import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        df = pd.read_csv(filepath)

        # 文字列映射成数字：male/female → 0/1，港口 S/C/Q → 0/1/2
        df['Sex'] = df['Sex'].map({'male':0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

        # Age 有缺失，用平均年龄补上；Embarked 缺 2 个，用 0（即 S，最常见）补
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Embarked'] = df['Embarked'].fillna(0)

        # 选 7 个数值特征做输入，Survived 做标签
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.len = df.shape[0]
        self.x_data = torch.from_numpy(df[features].values.astype(np.float32))
        self.y_data = torch.from_numpy(df[['Survived']].values.astype(np.float32))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

DATA_PATH = r'D:\Python_code\Kaggle_Titanic\titanic\train.csv'
dataset = DiabetesDataset(DATA_PATH)
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)   # 原来是 2：Windows 上每个 epoch 都会重开两个子进程、各自再 import 一次 torch，内存不够就崩（WinError 1455）；442 条数据用 0 完全够

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7,5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 课程里size_average=True 是旧写法，改用新写法
criterion = torch.nn.BCELoss(reduction='mean')   # size_average=True 是旧写法，等价于 reduction='mean'
# 0.1的准确率比0.01高
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# 不能直接循环训练，不然会报错，包装成main函数训练
if __name__ == '__main__':
    for epoch in range(1000):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    # 训练结束后，用整个数据集检查模型的分类准确率
    with torch.no_grad():                                  # 只做前向、不建计算图，省内存
        predicted = (model(dataset.x_data) > 0.5).float()  # 输出概率 > 0.5 判成 1，否则判成 0
        accuracy = (predicted == dataset.y_data).float().mean()
    print('训练集准确率 = %.2f%%' % (accuracy.item() * 100))