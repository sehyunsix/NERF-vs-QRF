import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *


class trainer:
    def __init__(
        self,
        model,
        data_loader,
    ):
        self.model = model
        self.data_loader = data_loader

    def process_in_batches(
        self,
        xy_coords,
        batch_size=64,
    ):

        all_predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(xy_coords), batch_size)):
                batch_coords = xy_coords[i : i + batch_size]
                # 모델을 사용하여 예측
                predictions = self.model(batch_coords)
                # 결과를 리스트에 저장
                all_predictions.append(predictions)  # GPU/MPS에서 CPU로 이동

        # 모든 배치를 합쳐서 하나의 텐서로 반환
        return torch.cat(all_predictions, dim=0)

    def train(self, num_epochs, lr):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # Mean Squared Error loss for RGB color prediction
        losses = []
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.data_loader):
                # Create a random batch
                xy_batch, color_batch = batch

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(xy_batch)
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
                        f"epoch: {epoch +1}/{num_epochs} step [{step+1}/{len(self.data_loader)}], Loss: {loss.item():.4f}"
                    )

        print("start predictions")
        # reate (x, y) grid for prediction
        xy_coords = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                xy_coords.append([x / WIDTH, y / HEIGHT])  # Normalize (x, y) to [0, 1]
                # xy_coords.append([x,y])

        xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
        print("proccessed batches")
        predictions = self.process_in_batches(xy_coords, batch_size=1024)
        predictions = predictions.detach().numpy()

        # Reshape predictions to image format
        predicted_img = predictions.reshape((HEIGHT, WIDTH, 3))  # (H, W, 3)

        # Convert to image format and save
        predicted_img = (predicted_img * 255).astype(
            np.uint8
        )  # Convert to uint8 [0, 255]
        img = Image.fromarray(predicted_img)
        return img
