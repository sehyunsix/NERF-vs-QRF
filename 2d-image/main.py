import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from model import MLPColorPredictor, PositionEncodingMLP
from q_model import QModel
import os
from tqdm import tqdm
from config import *
import pandas as pd
from datetime import datetime


def process_in_batches(
    xy_coords,
    model,
    batch_size=64,
):

    all_predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(xy_coords), batch_size)):
            batch_coords = xy_coords[i : i + batch_size]
            # 모델을 사용하여 예측
            predictions = model(batch_coords)
            # 결과를 리스트에 저장
            all_predictions.append(predictions)  # GPU/MPS에서 CPU로 이동

    # 모든 배치를 합쳐서 하나의 텐서로 반환
    return torch.cat(all_predictions, dim=0)


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


def calculate_psnr(img1, img2):
    mse = np.abs(np.mean((np.array(img1) / 255.0 - np.array(img2) / 255.0) ** 2))
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# 3. Training loop
def train_model(
    model,
    dataloader,
    xy_coords,
    colors,
    num_epochs=1000,
    batch_size=1024,
    lr=1e-4,
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
    return img


# Example usage
results_df = []


# Define the layers and methods to test
layers = [3, 4, 5]
methods = ["MLP", "PQC"]

# Iterate over methods and layers
for method in methods:
    for layer in layers:
        # Update the model based on the method and layer
        if method == "MLP":
            model = MLPColorPredictor(layer)
        elif method == "PQC":
            model = QModel(layer)

        # Train the model and get PSNR for each image
        psnr_list = []
        for i in range(3):
            input_img = f"test{i+1}.jpg"
            xy_coords, colors = load_image(IMAGE_PATH + input_img)
            dataloader = create_dataloader(
                xy_coords=xy_coords, colors=colors, batch_size=BATCH_SIZE, shuffle=True
            )
            img = train_model(
                model,
                dataloader,
                xy_coords,
                colors,
                num_epochs=EPOCH,
                batch_size=BATCH_SIZE,
                lr=LR,
            )
            img_name = f"{datetime.now()}_{layer}_{method}_test_image{i}"
            output_path = os.path.join("results/" + IMAGE_PATH, f"{img_name}.png")
            img.save(output_path)
            label_img = Image.open(IMAGE_PATH + input_img).convert("RGB")
            psnr = calculate_psnr(img, label_img)
            psnr_list.append(psnr)

        # Append the results to the DataFrame
        results_df.append(
            {
                "Layers": f"{method} ({layer} layer{'s' if layer > 1 else ''})",
                "Image 1": psnr_list[0],
                "Image 2": psnr_list[1],
                "Image 3": psnr_list[2],
            }
        )
results_df = pd.DataFrame(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv("results_psnr.csv", index=False)
print("Results saved to results_psnr.csv")
