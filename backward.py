import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])
# 开启这个才能计算梯度
w.requires_grad = True

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("predict (before training)", 4, forward(4).item())

# 注意grad是tensor变量，会构建计算图，额外占用内存，所以要加后缀
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data = w.data - 0.01*w.grad.data
        # 梯度数据要清零，不清零就是上一次计算与这一次计算清零，在backward内部累加
        w.grad.data.zero_()

    print("process:",epoch, l.item())

print("predict (after training)", 4, forward(4).item())