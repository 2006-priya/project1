import torch
import torch.optim as optim
from tqdm import tqdm

from model import VGGFeatures, gram_matrix
from utils import load_image, save_image
import config


def train(content_path, style_path, output_path):

    device = config.DEVICE

    # Load images
    content = load_image(content_path, config.IMAGE_SIZE).to(device)
    style = load_image(style_path, config.IMAGE_SIZE).to(device)

    # Load model
    model = VGGFeatures().to(device).eval()

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Generated image
    generated = content.clone().requires_grad_(True)

    # Optimizer
    optimizer = optim.Adam([generated], lr=config.LEARNING_RATE)

    # Precompute content & style features (optimization improvement)
    content_features = model(content)
    style_features = model(style)

    for step in tqdm(range(config.NUM_STEPS)):

        gen_features = model(generated)

        # -------- Content Loss --------
        content_loss = torch.mean(
            (gen_features[config.CONTENT_LAYER] -
             content_features[config.CONTENT_LAYER]) ** 2
        )

        # -------- Style Loss --------
        style_loss = 0

        for layer in config.STYLE_LAYERS:
            gen_feature = gen_features[layer]
            style_feature = style_features[layer]

            G = gram_matrix(gen_feature)
            A = gram_matrix(style_feature)

            # Print Gram matrix shape only once
            if step == 0:
                print(f"\nLayer: {layer}")
                print("Generated Gram shape:", G.shape)
                print("Style Gram shape:", A.shape)

            _, c, h, w = gen_feature.shape
            style_loss += torch.mean((G - A) ** 2) / (c * h * w)

        # -------- Total Loss --------
        total_loss = (
            config.CONTENT_WEIGHT * content_loss +
            config.STYLE_WEIGHT * style_loss
        )

        # -------- Backprop --------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Clamp image values (important)
        with torch.no_grad():
            generated.clamp_(0, 1)

        # -------- Print Loss --------
        if step % 50 == 0:
            print(f"\nStep {step}")
            print(f"Content Loss: {content_loss.item():.4f}")
            print(f"Style Loss: {style_loss.item():.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")

    # Save final image
    save_image(generated.detach(), output_path)


if __name__ == "__main__":
    train(
        "C:/Users/Dharsha/Downloads/pic1.jpg",
        "C:/Users/Dharsha/Downloads/pic2.jpg",
        "C:/Users/Dharsha/Downloads/pic3.png"
    )
