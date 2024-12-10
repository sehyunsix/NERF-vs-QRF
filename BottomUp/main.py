import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

# Numpy
import numpy as np

# Plot
import matplotlib.pyplot as plt


# User-Defined Class
from model import sin_ml, sin_qml
from trainer import trainer


def make_data_loader(func):

    # 데이터 생성
    num_data = 20000  # 총 데이터 개수
    batch_size = 128  # 배치 크기

    # 입력 데이터 생성 (0부터 2π 사이의 값)
    x_data = torch.rand(num_data, 1)  # (20000, 1) shape로 생성
    if "sin" == func:
        y_data = torch.sin(x_data)
    elif "tanh" == func:
        y_data = torch.tanh(x_data)
    elif "x" == func:
        y_data = x_data
    else:
        return

    # 텐서 데이터셋 생성
    dataset = TensorDataset(x_data, y_data)

    # Train/Test set 분할 (80%/20% 비율)
    train_size = int(0.8 * num_data)
    test_size = num_data - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


results_df = []
method_list = ["MLP", "PQC"]
func_list = ["sin", "tanh", "x"]
layer_num = [3, 4, 5]
for method in method_list:
    for i in range(3):
        mse_list = []
        for func in func_list:
            train_loader, test_loader = make_data_loader(func)
            if method == "PQC":
                model = sin_qml(num_qubit=3, num_layer=layer_num[i])

            else:
                model = sin_ml(num_layer=layer_num[i])
            trainer1 = trainer(
                model, train_loader=train_loader, test_loader=test_loader
            )
            trainer1.train(epochs=50)
            test_loss_list = trainer1.test()
            mse = np.array(test_loss_list).mean()
            mse_list.append(round(mse, 3))
        results_df.append(
            {
                "Layers": f"{method} ({layer_num[i]} layer{'s' if layer_num[i] > 1 else ''})",
                "sin": mse_list[0],
                "tanh": mse_list[1],
                "x": mse_list[2],
            }
        )
results_df = pd.DataFrame(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv("results_mse.csv", index=False)
print("Results saved to results_mse.csv")
