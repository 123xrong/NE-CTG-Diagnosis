import argparse
import torch
import wandb
from load_and_preprocess_data import load_and_preprocess_data
from models import CNN, EnhancedCNN, CNN_LSTM, CNN1D, EnhancedCNN1D, TimeSeriesTransformer
from evaluate import train_model
from evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_model(model_type, input_size, features, d_model, seq_len):
    if model_type == 'cnn':
        return CNN(input_size=input_size)
    elif model_type == 'enhanced_cnn':
        return EnhancedCNN(input_size=input_size)
    elif model_type == 'cnn_lstm':
        return CNN_LSTM(input_size=input_size)
    elif model_type == 'cnn1d':
        return CNN1D(input_size=input_size)
    elif model_type == 'enhanced_cnn1d':
        return EnhancedCNN1D(input_size=input_size)
    elif model_type == 'transformer':
        return TimeSeriesTransformer(input_size=len(features), num_heads=4, num_layers=2, d_model=d_model, seq_len=seq_len)

def setup_fine_tuning(model, fine_tuning_type):
    if fine_tuning_type == 'full_ft':
        # All parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
    elif fine_tuning_type == 'linear_probing':
        # Freeze all parameters except for the classifier layer
        for name, param in model.named_parameters():
            param.requires_grad = 'classifier' in name
    elif fine_tuning_type == 'lora':
        # Only train the LoRA layers and the classifier layer
        for name, param in model.named_parameters():
            param.requires_grad = 'lora_A' in name or 'lora_B' in name or 'classifier' in name
    else:
        raise ValueError(f"Unsupported fine tuning type: {fine_tuning_type}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model on time series data')
    parser.add_argument('--input_size', type=int, required=True, help='Size of the input time series data (e.g., number of time steps)')
    parser.add_argument('--public_data_path', type=str, required=True, help='Path to the public data file for pretraining')
    parser.add_argument('--private_data_path', type=str, required=True, help='Path to the private data file for fine-tuning')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'enhanced_cnn', 'cnn_lstm', 'cnn1d', 'enhanced_cnn1d', 'ffn', 'transformer'], required=True, help='Choose the model type: cnn, cnn_lstm, cnn1d, ffn, or transformer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--features', nargs='+', default=['FHR1', 'TOCO'], help='Features to use for training (e.g., FHR1, TOCO)')
    parser.add_argument('--use_segmentation', action='store_true', help='If set, will segment the data into windows of size 200')
    parser.add_argument('--window_size', type=int, default=200, help='The window size for data segmentation')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension for the Transformer (default: 64)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for the Transformer (default: 4)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers (default: 2)')
    parser.add_argument('--num_channels', type=int, default=2, help='Number of channels of the input (default: 2)')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of epochs to pretrain the model (default: 10)')
    parser.add_argument('--fine_tuning_type', type=str, choices=['full_ft', 'linear_probing', 'lora'], required=True, help='Choose the fine-tuning method: full fine-tuning, linear probing, or LoRA')

    args = parser.parse_args()

    # Load and preprocess data (with or without segmentation)
    # Load and preprocess public data
    public_X_train, public_X_test, public_y_train, public_y_test = load_and_preprocess_data(
        args.public_data_path, features_to_use=args.features, use_segmentation=args.use_segmentation, window_size=args.window_size,
    )

    # Load and preprocess private data
    private_X_train, private_X_test, private_y_train, private_y_test = load_and_preprocess_data(
        args.private_data_path, features_to_use=args.features, use_segmentation=args.use_segmentation, window_size=args.window_size,
    )

    # Create DataLoaders for public and private datasets
    public_train_loader = DataLoader(TensorDataset(public_X_train, public_y_train), batch_size=args.batch_size, shuffle=True)
    public_test_loader = DataLoader(TensorDataset(public_X_test, public_y_test), batch_size=args.batch_size, shuffle=False)  # For evaluating pretraining

    private_train_loader = DataLoader(TensorDataset(private_X_train, private_y_train), batch_size=args.batch_size, shuffle=True)
    private_test_loader = DataLoader(TensorDataset(private_X_test, private_y_test), batch_size=args.batch_size, shuffle=False)  # For evaluating fine-tuning

    # Define model. Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_type, args.input_size, args.features, args.d_model, args.input_size).to(device)
    
    # Pretraining
    with wandb.init(project="CTG", entity="your_username", name=f"{args.model_type}_pretraining", config=args):
        print(f"Pretraining {args.model_type} on public dataset for {args.pretrain_epochs} epochs")
        train_model(model, public_train_loader, public_test_loader, args.input_size, device, num_epochs=args.pretrain_epochs, phase='Pretraining')
        torch.save(model.state_dict(), 'pretrained_model.pth')

    # Load pretrained model and setup for fine-tuning
    model.load_state_dict(torch.load('pretrained_model.pth'))
    setup_fine_tuning(model, args.fine_tuning_type)

    # Fine-tuning
    with wandb.init(project="CTG", entity="your_username", name=f"{args.model_type}_fine-tuning", config=args):
        print(f"Fine-tuning {args.model_type} on private dataset for {args.epochs} epochs")
        train_model(model, private_train_loader, private_test_loader, args.input_size, device, num_epochs=args.epochs)

    # Evaluation and Logging
    eval_metrics = evaluate_model(model, private_test_loader, device)
    print(f"Evaluating {args.model_type} on private test set")
    print(eval_metrics)  # Output eval metrics to console

    # Log evaluation metrics to wandb
    wandb.init(project="CTG", entity="your_username", name=f"{args.model_type}_evaluation", config=args)
    wandb.log(eval_metrics)
    wandb.finish()

if __name__ == "__main__":
    main()