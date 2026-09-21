# 模型为 y = w1*(x**2) + w2*x + b
import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = torch.Tensor([1.0])
w1.requires_grad = True

w2 = torch.Tensor([1.0])
w2.requires_grad = True

b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1*(x**2) + w2*x + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("predict (before training)", 4, forward(4).item())

for epoch in range(5000):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("process:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())