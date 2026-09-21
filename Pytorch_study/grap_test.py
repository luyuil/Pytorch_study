import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    return (forward(x, w, b) - y) ** 2

w_list = np.arange(0.0,4.1,0.1)
b_list = np.arange(-2.0,2.1,0.1)
mse_list = []

# 关键：meshgrid 生成二维网格
W, B = np.meshgrid(w_list, b_list)   # 默认 indexing='xy'
# W.shape = (len(b_list), len(w_list))

# 计算每个 (w, b) 组合的 MSE
mse_grid = np.zeros_like(W)

for i in range(W.shape[0]):        # 遍历 b
    for j in range(W.shape[1]):    # 遍历 w
        w_val = W[i, j]
        b_val = B[i, j]
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w_val, b_val)
        mse_grid[i, j] = l_sum / 3

# 绘制 3D 曲面
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, B, mse_grid, cmap='viridis', alpha=0.8)

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Loss Surface: y = x*w + b')

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
