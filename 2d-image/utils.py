from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch


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


def calculate_psnr(img1, img2):
    mse = np.abs(np.mean((np.array(img1) / 255.0 - np.array(img2) / 255.0) ** 2))
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
