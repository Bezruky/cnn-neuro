import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

# Установка Seed для воспроизводимости
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузим предварительно обученную модель ResNet18 с параметром для более быстрой обучаемости
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Заменим последний слой (fully connected) так, чтобы количество выходных каналов соответствовало 5 классам цветов
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Определим трансформации
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_dir = "dataset/train"
dataset = ImageFolder(dataset_dir, transform=train_transforms)

print(dataset.classes)
exit()

# Разделяем фото в dataset на train и validate
train_size = int(0.8 * len(dataset))  # 80% - обучение
validate_size = len(dataset) - train_size  # 20% - валидация
train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])


# Создадим датагенераторы
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

# Определим функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #  lr=0.0005

# число эпох
num_epochs = 10

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_accuracy = 0
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Валидация модели
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Эпоха [{epoch+1}/{num_epochs}], '
          f'Потеря при обучении: {train_loss:.4f}, Точность при обучении: {train_accuracy:.4f}, '
          f'Потеря при валидации: {val_loss:.4f}, Точность при валидации: {val_accuracy:.4f}')

    # Сохранение лучшей модели на основе валидационной точности
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print('Сохранена лучшая модель!')
    
    # Сохранение последней актуальной модели
    torch.save(model.state_dict(), 'last_model.pth')


# Построим графики
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Обучение')
plt.plot(range(1, num_epochs+1), val_losses, label='Валидация')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.title('Потеря к эпохе')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Обучение')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Валидация')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('Точность к эпохе')
plt.legend()

plt.show()
