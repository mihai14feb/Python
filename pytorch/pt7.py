import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Date
transform = transforms.ToTensor() # transforma intr-un tensor de forma (1, 28, 28) (1 = canal alb negrum, 28x28 pixeli)
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # descarca si incarca setul MNIST pentru antrenare, aplica transformarea si descarca
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # grupeaza datele in cate 64 de imagini, le schimba ordinea sa nu memoreze modelul ordinea

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) # transforma (1, 28, 28) in (16, 26, 26) (2px se pierd din cauza la padding), se adauga un filtru 3x3
        self.pool = nn.MaxPool2d(2, 2) # face imaginea de 2 ori mai mica, o transforma in (16, 13, 13)
        self.conv2 = nn.Conv2d(16, 32, 3) # transforma in (32, 11, 11)
        self.fc1 = nn.Linear(32*5*5, 128) # primul strat linear, imaginea devine 5x5 cu 32 canale, deci 32*5*5, iese 128
        self.fc2 = nn.Linear(128, 10) # stratul final, raman 10 cifre (0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # primul strat, imaginea e scanata cu 16 luoe si simplificata
        x = self.pool(self.relu(self.conv2(x))) # alte 32 de lupe
        x = x.view(-1, 32*5*5) # intinde dimensiunea tensorului
        x = self.relu(self.fc1(x)) # functie liniara
        x = self.fc2(x) # ultima functie liniara care lasa cifrele 0-9
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in train_loader: # itereaza 64 de imagini
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoca {epoch+1}, Loss {loss.item()}")
