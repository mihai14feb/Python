import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2,4) # primul layer, 2 intrari, 4 iesiri
        self.layer2 = nn.Linear(4,1) # al doilea layer, 4 intrari, o iesire
        self.relu = nn.ReLU() # pregatim functia relu (daca dupa prelucrare nr e negativ, va fi transformat in 0
        self.sigmoid = nn.Sigmoid() # pregatim functia sigmoid (transforma nr in nr din intervalul [0,1]

    def forward(self, x):
        x = self.relu(self.layer1(x)) # aplicam functia relu
        x = self.sigmoid(self.layer2(x)) # aplicam functia sigmoid
        return x

model = XORModel()
criterion = nn.BCELoss() # alt tip de pierdere
optimizer = optim.Adam(model.parameters(), lr=0.01) # optimizarea Adam e mai eficienta decat SGD

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print(model(X).round()) # round aproximeaza