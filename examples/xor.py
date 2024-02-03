import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ai3

torch.manual_seed(0)

X = torch.Tensor([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
Y = torch.Tensor([0, 1, 1, 0])


class XOR(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x


model = XOR()
model = ai3.optimize(model)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)


weights_init(model)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

epochs = 2000
steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        x_var = Variable(X[j], requires_grad=False)
        y_var = Variable(Y[j], requires_grad=False)

        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()

        if i % 500 == 0 and j == 0:
            print("Epoch: {}, Loss: {}, ".format(i, loss.data))
