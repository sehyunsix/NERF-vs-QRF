
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import random
from model import MLPColorPredictor, PositionEncodingMLP
import os

def save_weights(model, epoch,image,method, folder='weights'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, f'model_{image}_{method}_{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

# 1. Load image and create (x, y) -> RGB dataset
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    height, width, _ = img.shape
    print("height, width",height,width)
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
def train_model(model, xy_coords, colors, num_epochs=1000, batch_size=1024, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error loss for RGB color prediction

    for epoch in range(num_epochs):
        # Create a random batch
        xy_batch, color_batch = create_batch(xy_coords, colors, batch_size)
        xy_batch=xy_batch.to('mps')
        color_batch=color_batch.to('mps')
        # Forward pass
        optimizer.zero_grad()
        predictions = model(xy_batch)

        # Compute loss
        loss = criterion(predictions, color_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage
image_path = 'test1.jpg'
method ='mlp' # Replace with the path to your image
xy_coords, colors = load_image(image_path)

# Initialize model
if method =="position":
  model = PositionEncodingMLP().to('mps')
else:
  model=MLPColorPredictor().to('mps')

# Train model
train_model(model, xy_coords, colors, num_epochs=1000, batch_size=128, lr=1e-4,)


save_weights(model, epoch=1000 ,image=image_path,method=method)