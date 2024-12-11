import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
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

    def plot_list(self, data, title=""):
        # 텐서 리스트를 numpy 배열로 변환
        loss_values = [e.item() for e in data]
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label="Plotting Data")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self, epochs=10, chk=False):
        self.train_loss_list = []
        for i in tqdm(range(epochs)):
            for idx, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.train_loss_list.append(loss.mean())
                loss.backward()
                self.optimizer.step()

        if chk:
            self.plot_list(self.train_loss_list, title="Train Loss")

    def test(self):
        self.test_loss_list = []
        for idx, (x, y) in enumerate(self.test_loader):
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.test_loss_list.append(loss.mean().detach())
        return self.test_loss_list
