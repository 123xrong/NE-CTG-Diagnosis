import argparse
import torch
from load_and_preprocess_data import load_and_preprocess_data
from models import CNN, EnhancedCNN, CNN_LSTM, CNN1D, EnhancedCNN1D, TimeSeriesTransformer 
from evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model on time series data')
    parser.add_argument('--input_size', type=int, required=True, help='Size of the input time series data (e.g., number of time steps)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file (pickle or numpy file containing X and y)')
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
    args = parser.parse_args()

    # Load and preprocess data (with or without segmentation)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data(
        args.data_path, features_to_use=args.features, use_segmentation=args.use_segmentation, window_size=args.window_size,
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # metrics_accumulator = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    
    
    # Model selection
    if args.model_type == 'cnn':
        model = CNN(input_size=args.input_size).to(device)
    if args.model_type == 'enhanced_cnn':
        model = EnhancedCNN(input_size=args.input_size).to(device)
    elif args.model_type == 'cnn_lstm':
        model = CNN_LSTM(input_size=args.input_size).to(device)
    elif args.model_type == 'cnn1d':
        model = CNN1D(input_size=args.input_size).to(device)
    elif args.model_type == 'enhanced_cnn1d':
        model = EnhancedCNN1D(input_size=args.input_size).to(device)
    elif args.model_type == 'transformer':
        # Transformer setup requires sequence length (input_size) and features
        seq_len = args.input_size
        print(f"Using Transformer with sequence length {seq_len}")
        model = TimeSeriesTransformer(input_size=len(args.features), num_heads=args.num_heads, num_layers=args.num_layers, d_model=args.d_model, seq_len=seq_len).to(device)

    # Train the model
    print(f"Training {args.model_type} for {args.epochs} epochs with batch size {args.batch_size}")
    mean_train_losses, std_train_losses, mean_val_losses, std_val_losses = train_model(
    model, train_loader, test_loader, args.input_size, device, num_epochs=args.epochs
)

    print(f"Evaluating {args.model_type} on test set")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
