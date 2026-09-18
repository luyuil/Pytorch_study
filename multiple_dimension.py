# 多维特征输入示例：用 sklearn 自带的糖尿病数据集做回归
#
# 数据位置：D:\Pytorch\Lib\site-packages\sklearn\datasets\data
#   diabetes_data.csv.gz   -> 442 个样本 x 10 个特征（已标准化，空格分隔）
#   diabetes_target.csv.gz -> 442 个标签（连续值 25 ~ 346，表示病情进展指数）
#
# 注意：这份数据不是课程里那份 diabetes.csv.gz（那份是 8 个特征 + 0/1 标签的二分类数据），
#       所以这里按回归来写：输入 10 维、输出 1 个连续值、损失用 MSELoss。
#       想完全跟课程一致，就需要把课程那份 diabetes.csv.gz 放到本文件同目录下。

import numpy as np
import torch

DATA_DIR = r'D:\Pytorch\Lib\site-packages\sklearn\datasets\data'

# np.loadtxt 会自动解压 .gz 文件；
# 这两个文件是空格分隔的，所以不能写 delimiter=','（课程那份才是逗号分隔）
X = np.loadtxt(DATA_DIR + r'\diabetes_data.csv.gz', dtype=np.float32)                   # (442, 10)
y = np.loadtxt(DATA_DIR + r'\diabetes_target.csv.gz', dtype=np.float32).reshape(-1, 1)  # (442, 1)

# 特征本身已经标准化过，不用再处理；
# 标签是 25 ~ 346 的大数，直接算 MSE 数值会很大、训练也不稳，
# 所以先把标签标准化成均值 0、标准差 1，最后再换算回原来的尺度。
y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

x_data = torch.from_numpy(X)          # (442, 10)
y_data = torch.from_numpy(y_scaled)   # (442, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 输入维度用 x_data.shape[1] 自动取（这里是 10），
        # 这样就不会再出现"写死 8 维、实际数据 10 维"导致的形状报错
        self.linear1 = torch.nn.Linear(x_data.shape[1], 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        # 最后一层不要加 sigmoid：回归任务的输出需要是任意实数
        # （标准化后的标签有正有负，被 sigmoid 压到 0~1 就学不了了）
        return self.linear3(x)


model = Model()

# 回归用 MSELoss；BCELoss 是分类用的，要求标签落在 0~1 之间
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5001):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 每 500 轮打印一次，避免刷屏
    if epoch % 500 == 0:
        print(epoch, round(loss.item(), 4))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 把预测结果换算回原始尺度，看看真实的误差有多大
with torch.no_grad():
    pred = model(x_data).numpy() * y_std + y_mean

mae = float(np.abs(pred - y).mean())
rmse = float(np.sqrt(((pred - y) ** 2).mean()))
print()
print('平均绝对误差 MAE  = %.1f' % mae)
print('均方根误差   RMSE = %.1f   (标签范围 %.0f ~ %.0f，标准差 %.1f)'
      % (rmse, y.min(), y.max(), y_std))
print('前 5 个样本的预测:', np.round(pred[:5, 0], 1).tolist())
print('前 5 个样本的真实:', np.round(y[:5, 0], 1).tolist())