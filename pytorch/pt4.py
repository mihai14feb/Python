import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2000.0],[2500.0],[3000.0],[3500.0]])

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[5.0]])).item())

plt.scatter(X, y, color='blue')
plt.plot(X, model(X).detach(), color='red')
plt.show()