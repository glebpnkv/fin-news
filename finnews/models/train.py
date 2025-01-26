import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from finnews.models.fc import FCBinaryClassifier


def train_fc_binary_classifier(
    model: FCBinaryClassifier,
    optimizer: optim,
    dataloader_train,
    dataloader_val = None,
    save_dir=None,
    epochs=10,
    device='cpu',
    patience=10,
):
    model.to(device)
    criterion = nn.BCELoss()
    best_roc_auc = 0  # Initialize best ROC AUC score
    epochs_no_improve = 0  # Initialize early stopping counter

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0.0

        # Training loop
        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            # Move data to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()  # Ensure output is squeezed for BCELoss
            loss = criterion(outputs, targets.float())  # Cast targets to float

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss_train += loss.item()

        # Calculate average training loss
        avg_train_loss = total_loss_train / len(dataloader_train)

        if dataloader_val:
            model.eval()  # Set model to evaluation mode
            total_loss_val = 0.0
            correct_predictions = 0
            total_samples = 0
            all_targets = []
            all_predictions = []

            with torch.no_grad():  # No gradient calculation during validation
                for inputs, targets in dataloader_val:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets.float())
                    total_loss_val += loss.item()

                    # Calculate validation accuracy
                    preds = (outputs >= 0.5).int()  # Threshold for binary classification
                    correct_predictions += (preds == targets).sum().item()
                    total_samples += targets.size(0)

                    # Collect targets and predictions for metrics calculation
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(outputs.cpu().numpy())

            # Calculate average validation loss and accuracy
            avg_val_loss = total_loss_val / len(dataloader_val)
            val_accuracy = correct_predictions / total_samples

            # Calculate metrics
            all_targets = np.array(all_targets)
            all_predictions = np.array(all_predictions)

            precision = precision_score(all_targets, (all_predictions >= 0.5).astype(int))
            recall = recall_score(all_targets, (all_predictions >= 0.5).astype(int))
            f1 = f1_score(all_targets, (all_predictions >= 0.5).astype(int))
            roc_auc = roc_auc_score(all_targets, all_predictions)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

            # Save model if ROC AUC improves
            if (roc_auc > best_roc_auc) and save_dir:
                best_roc_auc = roc_auc
                # Save model to specified directory
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "best_model.pt")
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        else:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss_train / len(dataloader_train):.4f}")
