import torch
from PIL import Image
import numpy as np
from model import MLPColorPredictor, PositionEncodingMLP
import os
from q_model import sin_qml
from tqdm import tqdm


# 2. 모델의 가중치를 불러오는 함수
def load_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    # model.eval()  # Set the model to evaluation mode
    print(f"Model weights loaded from {weight_path}")


def process_in_batches(
    xy_coords,
    model,
    batch_size=64,
):
    # 입력 데이터를 텐서로 변환xw
    # 모델을 사용할 장치로 이동 (MPS는 Mac에서 Metal 가속 장치)

    # 모델 예측을 저장할 리스트
    all_predictions = []

    # 데이터를 batch 단위로 나누어 처리
    # The `with torch.no_grad():` statement is a context manager provided by PyTorch
    # that temporarily sets all the `requires_grad` flags to `False`. This means that
    # any operations inside this context block will not build the computational graph
    # for gradient computation. It is commonly used during inference or evaluation to
    # disable gradient calculation, which can help reduce memory consumption and speed
    # up computations since no gradients need to be calculated or stored.
    with torch.no_grad():
        for i in tqdm(range(0, len(xy_coords), batch_size)):
            batch_coords = xy_coords[i : i + batch_size]
            # 모델을 사용하여 예측
            predictions = model(batch_coords)
            # 결과를 리스트에 저장
            all_predictions.append(predictions)  # GPU/MPS에서 CPU로 이동

    # 모든 배치를 합쳐서 하나의 텐서로 반환
    return torch.cat(all_predictions, dim=0)


# 3. 예측된 이미지를 result 폴더에 저장하는 함수
def save_predicted_image(model, width, height, image, method, output_folder="results"):
    print("start predictions")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create (x, y) grid for prediction
    xy_coords = []
    for y in range(height):
        for x in range(width):
            xy_coords.append([x / width, y / height])  # Normalize (x, y) to [0, 1]
            # xy_coords.append([x,y])

    xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
    # xy_coords = xy_coords.to("mps")
    # Predict RGB values for each (x, y) coordinate
    # with torch.no_grad():
    # model.to("mps")
    print("proccessed batches")
    predictions = process_in_batches(xy_coords, model, batch_size=64)
    predictions = predictions.detach().numpy()

    # Reshape predictions to image format
    predicted_img = predictions.reshape((height, width, 3))  # (H, W, 3)

    # Convert to image format and save
    predicted_img = (predicted_img * 255).astype(np.uint8)  # Convert to uint8 [0, 255]
    img = Image.fromarray(predicted_img)
    output_path = os.path.join(output_folder, f"predicted_{image}_{method}.png")
    img.save(output_path)
    print(f"Predicted image saved to {output_path}")


if __name__ == "__main__":
    image = "test1.jpg"
    method = "quantum"
    weight_path = "weights/model_epoch_1000.pth"
    weight_path = f"weights/model_{image}_{method}_1000.pth"

    load_weights(model, weight_path)
    if method == "mlp":
        model = MLPColorPredictor()
    elif method == "quantum":
        model = sin_qml(num_qubit=3, num_layer=3, use_qrelu=True)
    else:
        model = PositionEncodingMLP()
    save_predicted_image(model, width=1050, height=1400, image=image, method=method)
