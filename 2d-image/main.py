import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from model import MLPColorPredictor, PositionEncodingMLP
from predict import save_predicted_image, process_in_batches
from q_model import sin_qml
import os
from tqdm import tqdm
from config import *


def save_weights(model, epoch, image, method, folder="weights"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f"model_{image}_{method}_{epoch}.pth")
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


class CustomDataset(Dataset):
    def __init__(self, xy_coords, colors):
        self.xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
        self.colors = torch.tensor(colors, dtype=torch.float32)

    def __len__(self):
        return len(self.xy_coords)

    def __getitem__(self, idx):
        return self.xy_coords[idx], self.colors[idx]


# Dataset을 생성하고 DataLoader로 배치 및 셔플 설정
def create_dataloader(xy_coords, colors, batch_size, shuffle=True):
    dataset = CustomDataset(xy_coords, colors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# 1. Load image and create (x, y) -> RGB dataset
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    height, width, _ = img.shape
    print("height, width", height, width)
    # Create (x, y) and RGB datasets
    xy_coords = []
    colors = []

    for y in range(height):
        for x in range(width):
            xy_coords.append([x / width, y / height])  # Normalize (x, y) to [0, 1]
            # xy_coords.append([x,y])
            colors.append(img[y, x])  # RGB color at (x, y)

    return np.array(xy_coords), np.array(colors)


# 2. Create batch dataset for training
def create_batch(xy_coords, colors, batch_size):
    indices = random.sample(range(len(xy_coords)), batch_size)
    xy_batch = torch.tensor(xy_coords[indices], dtype=torch.float32)
    color_batch = torch.tensor(colors[indices], dtype=torch.float32)
    return xy_batch, color_batch


# 3. Training loop
def train_model(
    model, dataloader, xy_coords, colors, num_epochs=1000, batch_size=1024, lr=1e-4
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error loss for RGB color prediction
    losses = []
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            # Create a random batch
            xy_batch, color_batch = batch

            # Forward pass
            optimizer.zero_grad()
            predictions = model(xy_batch)
            # print("Pred :", predictions.shape)

            # Compute loss
            loss = criterion(predictions.float(), color_batch.float())
            # print(loss.dtype)
            losses.append(loss.item())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(
                    f"epoch: {epoch +1}/{EPOCH} step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss_graph_{METHOD}.png")  # Save the plot as an image file
    print("start predictions")

    # Create (x, y) grid for prediction
    xy_coords = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            xy_coords.append([x / WIDTH, y / HEIGHT])  # Normalize (x, y) to [0, 1]
            # xy_coords.append([x,y])

    xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
    print("proccessed batches")
    predictions = process_in_batches(xy_coords, model, batch_size=1024)
    predictions = predictions.detach().numpy()

    # Reshape predictions to image format
    predicted_img = predictions.reshape((HEIGHT, WIDTH, 3))  # (H, W, 3)

    # Convert to image format and save
    predicted_img = (predicted_img * 255).astype(np.uint8)  # Convert to uint8 [0, 255]
    img = Image.fromarray(predicted_img)
    output_path = os.path.join("results", f"predicted_{IMAGE_PATH}_{METHOD}.png")
    img.save(output_path)
    print(f"Predicted image saved to {output_path}")


# Example usage

xy_coords, colors = load_image(IMAGE_PATH)
dataloader = create_dataloader(
    xy_coords=xy_coords, colors=colors, batch_size=BATCH_SIZE, shuffle=True
)
# Initialize model
if METHOD == "position":
    model = PositionEncodingMLP()
elif METHOD == "quantum":
    model = sin_qml(WIRE, LAYER, True)
else:
    model = MLPColorPredictor()

# Train model
train_model(
    model,
    dataloader,
    xy_coords,
    colors,
    num_epochs=EPOCH,
    batch_size=BATCH_SIZE,
    lr=LR,
)


# save_weights(model, epoch=10, image=image_path, method=method)
