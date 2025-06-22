import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def reset_weights(m):
    """ Function to reset model weights to avoid weight leakage between runs. """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        m.reset_parameters()

def safe_convert_to_array(data_lists):
    """ Convert lists of different lengths to a numpy array by padding shorter lists with np.nan. """
    max_length = max(len(sublist) for sublist in data_lists)
    padded_lists = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in data_lists]
    return np.array(padded_lists)

def train_model(model, train_loader, device, num_epochs=30, num_runs=1, lr=0.001, step_size=10, gamma=0.1, phase='Pretraining'):
    criterion = torch.nn.CrossEntropyLoss()
    train_loss_lists = []

    for run in range(num_runs):
        logging.info(f"Starting training run {run + 1}/{num_runs}")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        train_loss_list = []

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss_list.append(avg_train_loss)
 # Initialize a new wandb run for each epoch
            wandb.log({f"{phase} Train Loss": avg_train_loss, "epoch": epoch})
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")

            scheduler.step()

        train_loss_lists.append(train_loss_list)
        model.apply(reset_weights)  # Reset weights for next training run

    train_loss_array = np.array(train_loss_lists)
    mean_train_losses = np.mean(train_loss_array, axis=0)
    std_train_losses = np.std(train_loss_array, axis=0)

    for epoch in range(len(mean_train_losses)):
        logging.info(f"Epoch {epoch + 1} - Mean Train Loss: {mean_train_losses[epoch]:.4f}, Std: {std_train_losses[epoch]:.4f}")

    return mean_train_losses, std_train_losses

def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    all_probabilities = []

    try:
        with torch.no_grad():  # Inference mode, no need to compute gradients
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_labels = torch.max(probs, dim=1)
                
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy()[:, 1])  # Assuming binary classification

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Log metrics to wandb
        wandb.log({'Evaluation Accuracy': accuracy,
                   'Evaluation Precision': precision,
                   'Evaluation Recall': recall,
                   'Evaluation F1 Score': f1})

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        accuracy = precision = recall = f1 = roc_auc = 0  # Default metrics in case of failure

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
