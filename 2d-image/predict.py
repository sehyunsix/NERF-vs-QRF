


import torch
from PIL import Image
import numpy as np
from model import MLPColorPredictor, PositionEncodingMLP
import os

# 2. 모델의 가중치를 불러오는 함수
def load_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path,weights_only=True))
    model.eval()  # Set the model to evaluation mode
    print(f"Model weights loaded from {weight_path}")

# 3. 예측된 이미지를 result 폴더에 저장하는 함수
def save_predicted_image(model, width, height,image,method, output_folder='results'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create (x, y) grid for prediction
    xy_coords = []
    for y in range(height):
        for x in range(width):
            xy_coords.append([x / width, y / height])  # Normalize (x, y) to [0, 1]
            # xy_coords.append([x,y])

    xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
    xy_coords=xy_coords.to('mps')
    # Predict RGB values for each (x, y) coordinate
    with torch.no_grad():
        model.to('mps')
        predictions = model(xy_coords)
        predictions= predictions.to('cpu').numpy()

    # Reshape predictions to image format
    predicted_img = predictions.reshape((height, width, 3))  # (H, W, 3)

    # Convert to image format and save
    predicted_img = (predicted_img * 255).astype(np.uint8)  # Convert to uint8 [0, 255]
    img = Image.fromarray(predicted_img)
    output_path = os.path.join(output_folder, f'predicted_{image}_{method}.png')
    img.save(output_path)
    print(f"Predicted image saved to {output_path}")

image='test1.jpg'
method='mlp'
weight_path = 'weights/model_epoch_1000.pth'
weight_path=f'weights/model_{image}_{method}_1000.pth'
if method =='mlp':
  model = MLPColorPredictor()
else:
  model=PositionEncodingMLP()
load_weights(model, weight_path)
save_predicted_image(model, width=1050, height=1400,image=image,method=method)