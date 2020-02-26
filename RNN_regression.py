import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible
# Hyper Parameters
TIME_STEP = 100      # rnn time step / image height
INPUT_SIZE = 3      # rnn input size / image width
LR = 0.02           # learning rate

#data
x1 = 20+torch.rand(100,1)*15
x2 = 6+torch.rand(100,1)*2
x3 = 100+torch.rand(100,1)*50
y=0.4+torch.rand(100,1)*0.5
x=torch.cat((x1, x2, x3), 1)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(  #
            input_size=3,
            hidden_size=64,     # rnn hidden unit
            num_layers=2,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out,(h_n, h_c)  = self.lstm(x, None)   # h_state 也要作为 RNN 的一个输入

        outs = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
loss_func = nn.MSELoss()

h_state = None   # 要使用初始 hidden state, 可以设成 None




for i in range(1000):
    b_x = x.view(-1, 100, 3)
    b_y = y.view(-1, 100, 1)

    # shape (batch, time_step, input_size)


    #output, h_state = rnn(b_x, h_state)   # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
    output= rnn(b_x)
    # !!  下一步十分重要 !!
    #h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错

    loss = loss_func(output, b_y)     # cross entropy loss
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()                    # apply gradients
    if (i + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))



# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01