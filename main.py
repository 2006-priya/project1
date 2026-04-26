import torch
import torch.optim as optim
from tqdm import tqdm

from model import VGGFeatures, gram_matrix
from utils import load_image, save_image
import config

def train(content_path, style_path, output_path):

    device = config.DEVICE

    content = load_image(content_path, config.IMAGE_SIZE).to(device)
    style = load_image(style_path, config.IMAGE_SIZE).to(device)

    model = VGGFeatures().to(device).eval()

    generated = content.clone().requires_grad_(True)

    optimizer = optim.Adam([generated], lr=config.LEARNING_RATE)

    for step in tqdm(range(config.NUM_STEPS)):

        gen_features = model(generated)
        content_features = model(content)
        style_features = model(style)

        # Content Loss
        content_loss = torch.mean(
            (gen_features[config.CONTENT_LAYER] -
             content_features[config.CONTENT_LAYER]) ** 2
        )

        # Style Loss
        style_loss = 0

        for layer in config.STYLE_LAYERS:
            gen_feature = gen_features[layer]
            style_feature = style_features[layer]

            G = gram_matrix(gen_feature)
            A = gram_matrix(style_feature)

            _, c, h, w = gen_feature.shape

            style_loss += torch.mean((G - A) ** 2) / (c * h * w)

        # Total Loss
        total_loss = (
            config.CONTENT_WEIGHT * content_loss +
            config.STYLE_WEIGHT * style_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Loss: {total_loss.item()}")

    save_image(generated, output_path)


if __name__ == "__main__":
    train(
        "C:/Users/Dharsha/Downloads/pic1.jpg",
        "C:/Users/Dharsha/Downloads/pic2.jpg",
        "C:/Users/Dharsha/Downloads/pic3.png"
    )