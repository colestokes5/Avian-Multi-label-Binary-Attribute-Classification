"""
File: attribute_detector.py
Author: Cole Stokes
Date: 2024-11-17
Last Modified: 2024-11-23
Description: Predicts the attributes that are present on a bird from an image using a CNN.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import LoadData, calculate_metrics, save_model


class ConvBlock(nn.Module):
    """
    Defines a convolution block with skip connections and max pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # Convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Skip connection and max pooling.
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.max_pool(self.conv_block(x) + self.skip(x))


class AttributeCNN(nn.Module):
    """
    Defines a convolution neural network.
    """
    def __init__(self, n_attributes=312):
        super(AttributeCNN, self).__init__()

        # Convolution blocks
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_attributes)
        )


    def forward(self, x):
        output = self.pool(self.conv_blocks(x))
        return self.fc(output.view(output.size(0), -1))


def train_cnn(lr=1e-4, epochs=25, batch_size=32):
    """
    Trains the convolution neural network, prints metrics, and saves the model.
    :param lr: float
    :param epochs: int
    :param batch_size: int
    """
    # Reads in the data and splits it 80% train and 20% validation.
    data = pd.read_csv("cub_200_2011.csv")
    train_data, val_data = train_test_split(data, train_size=0.8, random_state=200)

    # Loads the data into data loaders.
    # Shuffles and batches the train data.
    train_loader = DataLoader(LoadData(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LoadData(val_data), batch_size=1, shuffle=False)

    # Creates a DataFrame to store the model's metrics.
    model_metrics = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy", "val_accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "lr"])

    # Creates a model object and defines the optimizer, learning rate scheduler, and loss function.
    model = AttributeCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    attributes_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

    # Defines the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Epoch loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Training loop
        for images, attributes in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, attributes = images.to(device), attributes.to(device)

            # Passes through model and calculates loss.
            outputs = model(images)
            loss = attributes_loss(outputs, attributes)

            # Backward pass with gradient clipping.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Adds statistics to compute train loss and accuracy.
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == attributes).sum().item()
            train_total += attributes.numel()

        # Updates the learning rate.
        scheduler.step()

        # Gets the train and val metrics and saves to the metrics DataFrame.
        train_accuracy = 100 * train_correct / train_total
        val_accuracy, precision, recall, f1, cm = calculate_metrics(model, val_loader, device)
        metrics = pd.Series([epoch + 1, train_loss, train_accuracy, val_accuracy, precision, recall, f1, cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3], scheduler.get_last_lr()[0]], index=model_metrics.columns)
        model_metrics = pd.concat([model_metrics, metrics.to_frame().T], ignore_index=True)

        # Prints the metrics.
        print(f"Epoch {epoch + 1}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Saves the model.
        save_model(model, optimizer, epoch + 1, train_loss)

    # Saves the metric DataFrame.
    model_metrics.to_csv("model_metrics.csv", index=False)
    print("Model training complete.")


if __name__ == "__main__":
    train_cnn()
