import torch
from torchvision import transforms
from PIL import Image

def load_image(path, size):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)