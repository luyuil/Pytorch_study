# RNN
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

inputs = torch.randn(seq_len, batch_size, input_size)      # 整条序列一次给它
hidden = torch.zeros(num_layers, batch_size, hidden_size)  # 注意形状多了num_layers

out, hidden = cell(inputs, hidden)

print('Output size:', out.shape)
print('Output:', out)
print('Hidden size:', hidden.shape)
print("Hidden:", hidden)
