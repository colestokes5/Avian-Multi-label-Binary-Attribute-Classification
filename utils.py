"""
File: utils.py
Author: Cole Stokes
Date: 2024-11-18
Last Modified: 2024-11-22
Description: Defines functions that are needed to transform/load data, calculate metrics, and save models.
"""

import ast
import cv2
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset


class LoadData(Dataset):
    """
    Loads data to be utilized in a DataLoader.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = Path(row["path"])
        attributes = torch.tensor(ast.literal_eval(row["attributes"]), dtype=torch.float)

        # Loads and preprocesses image.
        image = preprocess_image(image_path, (64, 64))

        return image, attributes


def preprocess_image(image_path, target_size, augment=False):
    """
    Uses OpenCV to preprocess images.
    """
    # Loads and resizes the image.
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, target_size)

    # Normalizes and converts to PyTorch format by converting BGR to RGB and transposing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return torch.tensor(image, dtype=torch.float)


def calculate_metrics(model, val_data, device, range=None):
    """
    Calculates the validation accuracy, precision, recall, F1 score, and confusion matrix.
    """
    model.eval()

    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, attributes in val_data:
            # Saves the data to device and passes it though the model.
            image, attributes = image.to(device), attributes.to(device)
            outputs = model(image)
            if range != None:
                outputs = outputs[0][range[0]:range[1]+1]

            # Adds statistics to compute metrics.
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == attributes).sum().item()
            val_total += attributes.numel()
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(attributes.cpu().numpy())

    # Flattens the lists for metric calculation.
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Computes the metrics based on the entire val data.
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    val_accuracy = 100 * val_correct / val_total

    return val_accuracy, precision, recall, f1, cm


def save_model(model, optimizer, epoch, epoch_loss, model_dir="models"):
    """
    Saves the model, optimizer, epoch, and loss.
    """
    # Gets model directory path.
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Saves the model.
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': epoch_loss
    }, f"{model_dir}/attribute_cnn_epoch_{epoch}.pth")
