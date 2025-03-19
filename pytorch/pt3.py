import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], requires_grad=False)

class LinearRegression(nn.Module): # clasa cu parinte nn.Module (necesar pentru torch)
    def __init__(self): # constructor
        super(LinearRegression, self).__init__() # constructor la parinte (nn.Module)
        self.linear = nn.Linear(1, 1) # adauga functia lineara

    def forward(self, x):
        return self.linear(x) # calculeaza functia lineara pentru tensor

model = LinearRegression() # preia clasa
criterion = nn.MSELoss() # calculeaza pierderea/eroarea
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # optimizeaza codul

for epoch in range(10000):
    optimizer.zero_grad() # reseteaza gradientul la 0
    outputs = model(X) # calculeaza formula
    loss = criterion(outputs, y) # calculeaza pierderea
    loss.backward() # calculeaza gradientul pierderii
    optimizer.step() # actualizeaza greutatile prin gradientul calculat

print(model(torch.tensor([[5.0]])).item()) # item() afiseaza valoarea numerica

plt.scatter(X, y, color='blue')
plt.plot(X, model(X).detach(), color='red') # detach scoate predictia tip tensor, necesar la matplotlib
plt.show()