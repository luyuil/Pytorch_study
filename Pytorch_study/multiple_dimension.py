import torch
import numpy as np

# ================= 只有这段数据加载和原来不同 =================
# 课程里的 diabetes.csv.gz 在这台机器上不存在，这里改用 sklearn 自带的糖尿病数据：
#   ...\sklearn\datasets\data\diabetes_data.csv.gz    -> 442 个样本 x 10 个特征（空格分隔，已标准化）
#   ...\sklearn\datasets\data\diabetes_target.csv.gz  -> 442 个连续标签（25 ~ 346）
# 为了让下面的模型和 BCELoss 能直接套用，这里做了三件事：
#   1) 取前 8 个特征（要全部 10 个的话，把模型第一层改成 torch.nn.Linear(10, 6)）
#   2) 特征除以各自的标准差（原数据被压在 ±0.1 之间，不放大 sigmoid 网络几乎学不动）
#   3) 标签按中位数切成 0/1，变成二分类任务（原来的 BCELoss 要求标签在 0~1）
DATA_DIR = r'D:\Pytorch\Lib\site-packages\sklearn\datasets\data'
_features = np.loadtxt(DATA_DIR + r'\diabetes_data.csv.gz', dtype=np.float32)[:, :8]
_target = np.loadtxt(DATA_DIR + r'\diabetes_target.csv.gz', dtype=np.float32)
_features = _features / _features.std(axis=0)
_labels = (_target > np.median(_target)).astype(np.float32).reshape(-1, 1)
xy = np.hstack([_features, _labels])
# ================= 数据加载结束，下面基本保持原样 =================

x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 原来这里是 reduction='sum'：442 个样本时它把梯度放大 442 倍，配合 lr=0.1 会直接发散
criterion = torch.nn.BCELoss(reduction='sum')
# 原来是 lr=0.1：换成缩放后的 sklearn 特征后，1.0 在 100 轮内收敛得比较好
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 下面这两行是我加的：原来的代码看不到训练效果，这里顺便算一下分类准确率
with torch.no_grad():
    acc = ((model(x_data) > 0.5).float() == y_data).float().mean().item()
print('训练集准确率 = %.1f%%' % (acc * 100))