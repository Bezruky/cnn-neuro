import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image


# Загрузим предварительно обученную модель ResNet18
model = models.resnet18()
# Заменим последний слой (fully connected) так, чтобы количество выходных каналов соответствовало 3 классам
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Загрузка весов модели
model.load_state_dict(torch.load('best_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Предобработка изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузите изображение для инференса
image_path = 'dataset/test/Image_4.jpg'  # Замените на путь к вашему изображению
image = Image.open(image_path)
image_tensor = preprocess(image)
image_tensor = image_tensor.unsqueeze(0).to(device)  # Добавляем размер батча

# Прогоните изображение через модель
with torch.no_grad():
    output = model(image_tensor)

# Примените softmax для получения вероятностей классов
probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()

# Отображение результатов
top_prob, top_class = torch.topk(probabilities, 1)
top_prob = top_prob.item()
top_class = top_class.item()

# Получить имя класса (важен порядок, как в директории)
class_names = sorted(os.listdir("dataset/train"))
class_name = class_names[top_class]

# Отобразить изображение
plt.imshow(image)
plt.axis('off')
plt.title(f'Предсказание: {class_name} ({top_prob*100:.2f}%)')
plt.show()

# Построить барплот
plt.figure(figsize=(10, 5))
bars = plt.bar(range(len(probabilities)), probabilities, color='blue')
plt.xlabel('Классы')
plt.ylabel('Веростность')
plt.title('Вероятность классов')
plt.xticks(range(len(probabilities)), class_names, rotation='vertical')

# Подсветить класс с наибольшей уверенностью
bars[top_class].set_color('red')

plt.tight_layout()
plt.show()