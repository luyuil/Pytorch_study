# PyTorch 深度学习实践 · 学习笔记仓库

跟随《PyTorch 深度学习实践》课程（刘二大人，B站公开课）系统学习 PyTorch，
每节课把课程代码亲手敲一遍、跑通，并在此基础上做扩展实战（Kaggle 泰坦尼克号）。

- 课程：https://www.bilibili.com/video/BV1Y7411d7Ys （PyTorch 深度学习实践）
- 运行环境：Windows + conda 环境（`D:\Pytorch\envs\pytorch`），CPU 训练
- 数据集：MNIST 在 `dataset/mnist/`，课程自带数据在 `Pytorch_study/data/`

## 目录结构

```
Python_code/
├── Pytorch_study/          # 课程代码（按课程顺序）
│   ├── linear_regression.py        # L2 线性模型
│   ├── gradient_descent.py         # L3 梯度下降（手写）
│   ├── stochastic_gradient_descent.py  # L3 随机梯度下降（手写）
│   ├── backward.py / backward_test.py  # L4 反向传播与计算图
│   ├── logistic_regression.py      # L6 逻辑斯蒂回归
│   ├── multiple_dimension.py       # L7 多维特征输入
│   ├── Diabetes.py                 # L8 Dataset/DataLoader 加载糖尿病数据
│   ├── multi-class classification problem.py  # L9 多分类（MNIST，全连接）
│   ├── convolutional layer.py      # L10 Conv2d 维度演示
│   ├── padding.py                  # L10 padding 演示
│   ├── MaxPool.py                  # L10 池化
│   ├── ConvNet.py                  # L11 CNN 训练 MNIST（含 GPU 版）
│   ├── RNN-Cell-1.py ~ RNN-Cell-5.py  # L12 循环神经网络五步走
│   ├── RNNCell.py                  # L13 RNN 分类器：姓氏猜国家（完整版）
│   ├── diabetes.csv.gz             # sklearn 糖尿病数据自制的课程格式数据
│   └── data/                       # L13 数据：names_train/test.csv.gz
├── Kaggle_Titanic/         # 实战：Kaggle 泰坦尼克号
│   ├── titanic_train.py            # 仿照 Diabetes.py 的 MLP 训练
│   ├── titanic_test.py             # 8:2 划分验证集，检查过拟合
│   ├── titanic_submit.py           # 全量训练 + 生成 Kaggle 提交文件
│   └── submission.csv              # 提交结果
└── dataset/mnist/          # MNIST 数据（已下载）
```

## 学习路线与要点

### 第一阶段：从零理解训练（L2–L5）

- **线性回归**：手写梯度下降 `w = w - lr * grad`，再手写随机梯度下降，
  体会到"一次用一个样本 vs 用全部样本"在收敛速度和震荡上的差别。
- **反向传播**：理解计算图——`loss.backward()` 沿图用链式法则自动求导，
  每个张量上的 `grad_fn` 就是这条路径的痕迹。
- 之后的一切模型，更新参数都是同一套组合拳：**`zero_grad() → backward() → step()`**。

### 第二阶段：PyTorch 基本件（L6–L8）

- **逻辑斯蒂回归**：Sigmoid 把线性输出压成 (0,1) 概率；二分类配 `BCELoss`
  （旧写法 `size_average=True` 等价于 `reduction='mean'`）。
- **多维特征输入**：8 特征 → 6 → 4 → 1 的 MLP，首次写 `nn.Module` 的 class 定义，
  理解 `super().__init__()` 和 `model.parameters()` 的参数登记机制。
- **Dataset / DataLoader**：把数据包成 `Dataset`（三件套 `__init__/__getitem__/__len__`），
  `DataLoader` 负责 batch、shuffle；本仓库的 `diabetes.csv.gz` 是用 sklearn
  自带糖尿病数据自制成的课程格式。

### 第三阶段：图像与 CNN（L9–L11）

- **多分类（MNIST 全连接版）**：10 类用 `CrossEntropyLoss`——注意它**内部自带
  softmax**，模型只输出 10 个原始分数即可，最后一层不要接激活；
  预测取 `torch.max(outputs, dim=1)` 的下标。`Normalize((0.1307,),(0.3081,))`
  是 MNIST 全集的均值/标准差。
- **卷积积木**：`Conv2d` 的核心心智模型——**每个输出通道 = 一个卷积核**
  （核深度恒等于输入通道数），`out_channels` 就是"要几个特征检测器"；
  尺寸公式 `N_out = (N_in + 2*padding - kernel)/stride + 1`；
  权重共享让 CNN 参数量比全连接少几个数量级。
- **CNN 训练 MNIST**：两层"卷积→ReLU→池化"后摊平接全连接，
  维度链 28→24→12→8→4，20通道×4×4=320 对齐 fc；
  训练搬上 GPU（模型搬一次、数据每个 batch 搬）。
  最终测试准确率约 98%~99%，高于全连接版的 96.9%。

### 第四阶段：序列与 RNN（L12–L13）

- **RNN 五步走（RNN-Cell-1 ~ 5）**，每一步解决一个疑问：
  1. `RNNCell`：手写时间循环，`hidden = cell(input, hidden)` 把新状态写回去
     （拼写错一个字母，记忆就永远是 0——真踩过）；
  2. `nn.RNN`：循环内置，一次喂整条序列，返回 out（每步）+ hidden（最后一步），
     且初始 hidden 形状多了 `num_layers` 维；
  3. one-hot 版 "hello→ohlol"：手推 one-hot × 矩阵 = 查表；
  4. 整条序列 + CrossEntropyLoss：标签合同 **输入 (N,C) → 标签 (N,)**；
  5. `Embedding` 版：输入从 one-hot 换成**整数编号**（LongTensor），
     Embedding 是 4 行×10 维的可训练查表，取代"手工 one-hot 乘矩阵"。
- **名字猜国家（L13 完整版）**：ASCII 编号 → padding → 按长度降序排序 →
  `pack_padded_sequence`（让 GRU 不吃 padding 的 0）→ 双向 GRU →
  两方向最终记忆 `torch.cat` 拼接 → 全连接出 18 国分数。
  gzip+csv 读数据、国家名→编号字典、准确率曲线绘制。

### 实战：Kaggle 泰坦尼克号

仿照 Diabetes.py 的写法对泰坦尼克数据做生存预测，完整走了一遍"真实数据"的坑：

- `np.loadtxt` 读不了带表头和文字列的 csv → 改用 `pandas.read_csv`；
  Sex/Embarked 手动映射成数字，Age/Fare 缺失值填充。
- 训练集准确率每次运行都不一样：来自**权重随机初始化 + shuffle**，
  固定 `torch.manual_seed()` 即可复现。
- `titanic_test.py` 做 8:2 训练/验证划分——训练集准确率虚高，
  验证集才是真实水平，两者差距就是过拟合信号。
- `titanic_submit.py` 用全量数据训练、同一套预处理（均值从训练集传给测试集）
  预测 test.csv，生成 Kaggle 提交文件 `submission.csv`。

## 踩坑记录（血的教训）

| 坑 | 教训 |
|---|---|
| `torch.utiles.data` | 模块名是 `utils`，没有 e |
| 绝对路径报错 | Windows 路径加 `r` 前缀，防 `\t` 被转义成制表符 |
| mat1/mat2 形状乘不了 | `Linear` 输入维度必须等于特征数（8 vs 7 惨案） |
| has no len() | `__getitem__/__len__` 忘了提到类级别，缩进进了 `__init__` |
| Windows 开 num_workers=2 内存爆（WinError 1455） | 每个子进程重新 import 一遍 torch；小数据用 0，图片任务用 2 + `persistent_workers=True` |
| MNIST 下载卡 96.9% 后报 corrupted | S3 源慢，手动下载 4 个 .gz 放进 `dataset/mnist/MNIST/raw/` 即可 |
| BCELoss 报 all elements between 0 and 1 | 二分类输出要过 sigmoid；多分类用 CrossEntropyLoss 且不要 sigmoid |
| multi-target not supported | 整条序列喂交叉熵时标签要摊平成 (N,)，别带 batch 维 |
| Embedding 报 FloatTensor | Embedding 只吃 LongTensor 编号，不吃 one-hot |
| 跨文件 import optimizer/criterion | 优化器绑定的参数是"别人的模型"，step() 更新不到本模型；训练配置就地定义 |
| pack_padded_sequence 报错 | 长度必须降序排序；上 GPU 后长度张量要留在 CPU |

## 心得

- 深度学习的"新手三件套"贯穿始终：**造数据张量（对齐形状）→ 定义模型（对齐维度）→
  三兄弟训练循环**。所有模型都是这三个动作的排列组合。
- 维度是 RNN/CNN 的一切：把每一步的输入输出形状写进注释里，报错先查形状。
- 课程是骨架，实战是血肉——把课程代码搬到新数据集上（Titanic），
  遇到的每个报错都比听懂十个概念更有用。
