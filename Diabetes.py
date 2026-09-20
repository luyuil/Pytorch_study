# 数据说明（本文件只改了这里的数据引用）
# ---------------------------------------------------------------------------
# 课程里用的 diabetes.csv.gz 在这台电脑上不存在，这里改用 sklearn 自带的糖尿病数据：
#   D:\Pytorch\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv.gz   (442 x 10)
#   D:\Pytorch\Lib\site-packages\sklearn\datasets\data\diabetes_target.csv.gz (442 个连续标签)
# 处理方式：取前 8 个特征、每列缩放到标准差 1，标签按中位数切成 0/1，拼成 9 列，
# 存成同目录下的 diabetes.csv.gz（逗号分隔：前 8 列是特征，最后一列是标签），
# 格式和课程那份完全一致。以后拿到课程原版，直接覆盖这个文件即可，代码不用改。
# ---------------------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# num_workers是要不要开子进程并行计算的意思
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)   # 原来是 2：Windows 上每个 epoch 都会重开两个子进程、各自再 import 一次 torch，内存不够就崩（WinError 1455）；442 条数据用 0 完全够

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 课程里size_average=True 是旧写法，改用新写法
criterion = torch.nn.BCELoss(reduction='mean')   # size_average=True 是旧写法，等价于 reduction='mean'
# 0.1的准确率比0.01高
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# 不能直接循环训练，不然会报错，包装成main函数训练
if __name__ == '__main__':
    for epoch in range(100):
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
