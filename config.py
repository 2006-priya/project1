import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 512   # low memory safe
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 100000

LEARNING_RATE = 0.003
NUM_STEPS = 300

STYLE_LAYERS = ['0', '5', '10', '19', '28']
CONTENT_LAYER = '21'