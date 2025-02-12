import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from finnews.models import FCBinaryClassifier, InformerFusionModel


def train_fc_binary_classifier(
    model: FCBinaryClassifier,
    optimizer: optim,
    dataloader_train,
    dataloader_val = None,
    save_dir: str = None,
    epochs: int = 10,
    device: str | None = None,
    patience: int=10,
):
    """
    Trains a fully connected binary classifier (FCBinaryClassifier) model using the specified optimizer, and
    training data. Allows optional validation and saving of the best model based on the
    ROC AUC score. Includes support for early stopping.

    :param model: Fully connected binary classifier to be trained
    :type model: FCBinaryClassifier
    :param optimizer: Optimizer for model parameter updates
    :type optimizer: torch.optim.Optimizer
    :param dataloader_train: Dataloader for training dataset
    :type dataloader_train: torch.utils.data.DataLoader
    :param dataloader_val: Optional dataloader for validation dataset
    :type dataloader_val: torch.utils.data.DataLoader or None
    :param save_dir: Directory to save the best performing model
    :type save_dir: str or None
    :param epochs: Number of epochs for training
    :type epochs: int
    :param device: Computing device to use for training ('cpu' or 'cuda')
    :type device: str or None
    :param patience: Number of consecutive epochs without improvement in
        validation performance before early stopping
    :type patience: int
    :return: None
    """
    if device is None:
        device = Accelerator().device

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

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, "
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


def train_informer_fusion(
    model: InformerFusionModel,
    optimizer,
    dataloader_train,
    dataloader_val = None,
    save_dir: str = None,
    epochs: int = 10,
    device: str | None = None,
    patience: int = 10,
):
    accelerator = Accelerator()
    if device is None:
        device = accelerator.device
    model.to(device)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader_train,
    )

    if dataloader_val:
        dataloader_val = accelerator.prepare(dataloader_val)

    mae_loss = torch.nn.L1Loss(reduction='mean')
    best_mae = torch.inf  # Initialize best MAE score
    epochs_no_improve = 0  # Initialize early stopping counter

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0.0

        # Training loop
        num_batches = 0  # Number of batches is random
        for batch_idx, batch in tqdm(enumerate(dataloader_train)):
            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                inputs_fusion=batch["past_articles"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            # Accumulate loss
            total_loss_train += loss.item()
            num_batches += 1

        # Calculate average training loss
        avg_train_loss = total_loss_train / num_batches

        if dataloader_val:
            model.eval()  # Set model to evaluation mode
            # Evaluation metrics
            total_loss_val = 0.0
            correct_signs = 0
            total_sign_targets = []
            total_sign_predictions = []

            num_batches = 0  # Number of batches is random
            total_samples = 0
            with torch.no_grad():  # No gradient calculation during validation
                for batch in tqdm(dataloader_val):
                    outputs = model.generate(
                        past_time_features=batch["past_time_features"].to(device),
                        past_values=batch["past_values"].to(device),
                        inputs_fusion=batch["past_articles"].to(device),
                        future_time_features=batch["future_time_features"].to(device),
                    )

                    # Median prediction for point forecasts
                    outputs_preds = torch.quantile(outputs["sequences"], q=0.5, dim=1)
                    total_loss_val += mae_loss(
                        batch["future_values"].to(device),
                        outputs_preds
                    )

                    # Signs of forecasts
                    outputs_signs = torch.quantile(
                        torch.sign(torch.sum(outputs["sequences"], axis=-1)),
                        q=0.5,
                        dim=-1
                    )
                    outputs_signs = (outputs_signs > 0).int()
                    true_signs = (torch.sign(torch.sum(batch["future_values"], axis=-1)) > 0).int()
                    correct_signs += (outputs_signs.cpu() == true_signs.cpu()).sum().item()

                    # Collect targets and predictions for metrics calculation
                    total_sign_targets.extend(true_signs.cpu().numpy())
                    total_sign_predictions.extend(outputs_signs.cpu().numpy())

                    num_batches += 1
                    total_samples += batch["future_values"].size(0)

            # Calculate average validation loss and accuracy
            avg_val_loss = total_loss_val / num_batches
            val_accuracy = correct_signs / total_samples

            # Calculate metrics
            all_targets = np.array(total_sign_targets)
            all_predictions = np.array(total_sign_predictions)

            precision = precision_score(all_targets, (all_predictions >= 0.5).astype(int))
            recall = recall_score(all_targets, (all_predictions >= 0.5).astype(int))
            f1 = f1_score(all_targets, (all_predictions >= 0.5).astype(int))
            roc_auc = roc_auc_score(all_targets, all_predictions)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, Val MAE Loss: {avg_val_loss:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

            # Save model if ROC AUC improves
            if (avg_val_loss < best_mae) and save_dir:
                best_mae = avg_val_loss
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
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}")
