from PIL import Image
import random
import os
from utils import *
from config import *
import pandas as pd
from datetime import datetime
from model import MLPColorPredictor, QModel
from trainer import trainer


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
            tr = trainer(
                model,
                dataloader,
            )
            img = tr.train(EPOCH, LR)
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
