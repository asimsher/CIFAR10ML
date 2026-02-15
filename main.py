import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import io
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from pydantic import BaseModel
import streamlit as st




classes = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']
class CifarClassifacation(nn.Module):
  def __init__(self):
     super().__init__()
     self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
     )
     self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 2 * 2, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
     )

  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarClassifacation()
model.load_state_dict(torch.load('model (2).pth', map_location=device))
model.to(device)
model.eval()

cifar_app = FastAPI()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])



st.title('CIFAR10 CLASSIFIER MODEL')
st.text('Загрузить изображение с цифрой, и модель попробует ее распознать')

cifar_image = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg', 'svg'])

if not cifar_image:
    st.info('Загрузите изображение')
else:
    st.image(cifar_image, caption='Загруженное изображение')

if st.button('Определите цифру'):
    try:
        data = cifar_image.read()
        img = Image.open(io.BytesIO(data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            result = pred.argmax(dim=1).item()
        st.success(f'Модель думает, что это цифрa: {classes[result]}')

    except Exception as e:
        st.exception(f'Error {str(e)}')
