import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[1,1],[2,2],[3,3]], dtype=torch.float32)
y = torch.tensor([[1],[1],[0],[0]], dtype=torch.float32)

class CircleModel(nn.Module):
    def __init__(self):
        super(CircleModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = CircleModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print(model(X).round())