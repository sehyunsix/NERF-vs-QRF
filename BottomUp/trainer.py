import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from model import sin_ml
from torch import optim
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, clear_output


class trainer:

    def __init__(
        self, model, train_loader, test_loader, criterion=nn.MSELoss(), lr=0.001
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.train_loss_list = []
        self.test_loss_list = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def plot_list(self, data):
        # 텐서 리스트를 numpy 배열로 변환
        loss_values = [e.item() for e in data]

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def ax_plot_pred(self, ax):
        """
        Function to plot the predictions in real time.
        x: Input data (could be images or features)
        pred: Predicted outputs from the model
        ax: The matplotlib axis to update the plot
        """
        # Clear the previous plot
        ax.clear()
        x = np.linspace(0, 2 * np.pi, 100)
        x = torch.tensor(x.reshape(-1, 1)).to(torch.float32)
        pred = self.model(x)
        # Assume `x` and `pred` are 1D or simple tensors that can be plotted
        ax.plot(
            x.cpu().detach().numpy(),
            pred.cpu().detach().numpy(),
            "r-",
            label="Prediction",
        )

    def plot_pred(self):
        my_x = np.linspace(0, 2 * np.pi, 100)
        my_x = torch.tensor(my_x.reshape(-1, 1)).to(torch.float32)
        my_y = self.model(my_x)
        self.plot_list(my_y)

    def train(self, epochs=100, chk=False):
        plt.ion()
        self.train_loss_list = []
        fig, ax = plt.subplots()
        ax.set_title("Real-time Prediction Update")
        for i in tqdm(range(epochs)):
            for idx, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.train_loss_list.append(loss.mean())
                loss.backward()
                self.optimizer.step()
                if chk:
                    plt.draw()
                    fig.canvas.draw()
                    display(fig)
                    self.ax_plot_pred(ax)
                    clear_output(wait=True)

        if chk:
            self.plot_list(self.train_loss_list)

    def test(self, chk=True):
        self.test_loss_list = []
        for idx, (x, y) in enumerate(self.test_loader):
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.test_loss_list.append(loss.mean())
        if chk:
            self.plot_list(self.test_loss_list)