import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

def reset_weights(m):
    """ Function to reset model weights to avoid weight leakage between runs. """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        m.reset_parameters()

def safe_convert_to_array(data_lists):
    """ Convert lists of different lengths to a numpy array by padding shorter lists with np.nan. """
    max_length = max(len(sublist) for sublist in data_lists)
    padded_lists = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in data_lists]
    return np.array(padded_lists)

def train_model(model, train_loader, test_loader, input_size, device, num_epochs=30, num_runs=1, lr=0.001, step_size=10, gamma=0.1, phase='Pretraining'):
    criterion = torch.nn.CrossEntropyLoss()
    train_loss_lists = []
    val_loss_lists = []

    for run in range(num_runs):
        logging.info(f"Starting training run {run + 1}/{num_runs}")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_val_loss = float('inf')
        early_stop_count = 0
        early_stop_patience = 10
        train_loss_list = []
        val_loss_list = []

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                y_batch = y_batch.squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_loss = 0.0
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    y_val = y_val.squeeze()
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)

            # Log the average losses to wandb
            wandb.log({f"{phase} Train Loss": avg_train_loss, f"{phase} Validation Loss": avg_val_loss, "epoch": epoch})

            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                torch.save(model.state_dict(), 'best_model.pth')
                # Optionally, you might want to log the best model in wandb as well
                wandb.save('best_model.pth')
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            scheduler.step()

        train_loss_lists.append(train_loss_list)
        val_loss_lists.append(val_loss_list)
        model.apply(reset_weights)  # Reset weights for next training run

    train_loss_array = safe_convert_to_array(train_loss_lists)
    val_loss_array = safe_convert_to_array(val_loss_lists)

    mean_train_losses = np.nanmean(train_loss_array, axis=0)
    std_train_losses = np.nanstd(train_loss_array, axis=0)
    mean_val_losses = np.nanmean(val_loss_array, axis=0)
    std_val_losses = np.nanstd(val_loss_array, axis=0)

    for epoch in range(len(mean_train_losses)):
        logging.info(f"Epoch {epoch + 1} - Mean Train Loss: {mean_train_losses[epoch]:.4f}, Std: {std_train_losses[epoch]:.4f}, Mean Val Loss: {mean_val_losses[epoch]:.4f}, Std: {std_val_losses[epoch]:.4f}")

    return mean_train_losses, std_train_losses, mean_val_losses, std_val_losses

def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []

    try:
        with torch.no_grad():  # Inference mode, no need to compute gradients
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_labels = torch.max(probs, dim=1)
                
                all_predictions.append(predicted_labels)
                all_labels.append(labels)

        all_predictions = torch.cat(all_predictions).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        # Log metrics to wandb
        wandb.log({'Evaluation Accuracy': accuracy,
                   'Evaluation Precision': precision,
                   'Evaluation Recall': recall,
                   'Evaluation F1 Score': f1})

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        accuracy = precision = recall = f1 = 0  # Default metrics in case of failure

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
