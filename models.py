import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Compute the size of the flattened layer dynamically
        self.flattened_size = self._get_flattened_size(input_size)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_flattened_size(self, input_size):
        """Helper function to calculate the flattened size after conv and pooling layers."""
        # Create a dummy input tensor (batch_size=1, channels=2, input_size)
        dummy_input = torch.randn(1, 2, input_size)
        
        # Pass the dummy input through the conv and pooling layers
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Return the flattened size
        return x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # After conv1 and pooling
        x = self.pool(torch.relu(self.conv2(x)))  # After conv2 and pooling
        
        # Dynamically calculate the flatten size instead of hard-coding it
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self, input_size):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Additional layer
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # Additional layer

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)  # Changed to AvgPool1d

        # Compute the size of the flattened layer dynamically
        self.flattened_size = self._get_flattened_size(input_size)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_flattened_size(self, input_size):
        """Helper function to calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 2, input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Process through additional conv layers
        x = self.pool(F.relu(self.conv4(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Additional conv layer
        x = self.pool(F.relu(self.conv4(x)))  # Additional conv layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes=2, lstm_hidden_size=64, lstm_layers=1):
        super(CNN_LSTM, self).__init__()
        
        # Kernel, stride, and padding setup
        kernel_size = 120  # To capture features with a receptive field size about 30 data points
        stride = 4
        padding = kernel_size // 2  # Padding to maintain dimensionality

        # Define the convolutional layers (reduced to three layers)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding)

        # Average pooling to reduce dimensions further
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM Layer setup
        self.flattened_size = self._get_flattened_size(input_size)
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)  # Adjust to match output from LSTM
        self.fc2 = nn.Linear(128, num_classes)  # Binary classification (Healthy/Unhealthy)

    def _get_flattened_size(self, input_size):
        """Calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 1, input_size)  # Simulate input in the expected shape
        # Pass through each layer
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print("Shape after conv3:", x.shape)
        # No need to flatten as LSTM expects 3D input, but we need the correct last dimension size
        return x.size(1)  # This gives the number of features per time-step to the LSTM

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.transpose(1, 2)  # Reshape for LSTM; needs (batch, seq_len, features)

        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]  # We use only the last LSTM output for classification

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        
        # Single input channel now (only FHR1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # 1 channel for FHR1
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size after convolutions and pooling layers
        self.flattened_size = self._get_flattened_size(input_size)
        
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Adjust based on the flattened size
        self.fc2 = nn.Linear(128, 2)  # Binary classification (Healthy/Unhealthy)

    def _get_flattened_size(self, input_size):
        """Helper function to calculate the flattened size after conv and pooling layers."""
        # Create a dummy input tensor (batch_size=1, channels=1, input_size)
        dummy_input = torch.randn(1, 1, input_size)
        
        # Pass the dummy input through the conv and pooling layers
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Return the flattened size
        return x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # After conv1 and pooling
        x = self.pool(torch.relu(self.conv2(x)))  # After conv2 and pooling
        
        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FFN(nn.Module):
    def __init__(self, input_size):
        super(FFN, self).__init__()
        
        # Hidden layers
        self.fc1 = nn.Linear(input_size, 128)  # Dynamically adjust based on input size
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer for 2 classes (binary classification)
        self.fc3 = nn.Linear(64, 2)  # 2 output units, for 2 classes

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the input directly
        x = x.reshape(x.size(0), -1)  # Flatten the input [batch_size, time_steps * features]
        
        # Pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here (raw logits), CrossEntropyLoss expects logits
        
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=2, d_model=64, seq_len=7200):
        super(TimeSeriesTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)  # Project input to model dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(64, 2)  # Final classification layer
        
    def forward(self, x):
        # Assume x is (batch_size, seq_len, input_size)
        print(f"Shape of input: {x.shape}")
        x = x.transpose(1, 2)
        print(f"Shape after transpose: {x.shape}")
        x = self.embedding(x)  # Project the input to d_model dimensions
        print(f"Shape after embedding: {x.shape}")

        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        print(f"Shape after permutation: {x.shape}")

        x = self.transformer(x)
        print(f"Shape after transformer: {x.shape}")

        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        x = x.flatten(start_dim=1)  # Flatten all sequence steps and d_model
        x = self.fc(x)
        return x

class EnhancedCNN1D(nn.Module):
    def __init__(self, input_size):
        super(EnhancedCNN1D, self).__init__()
        
        # Kernel, stride, and padding setup
        kernel_size = 120  # To capture features with a receptive field size about 30 data points
        stride = 4
        padding = kernel_size // 2  # Padding to maintain dimensionality

        # Define the convolutional layers (reduced to three layers)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding)

        # Average pooling to reduce dimensions further
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        # Calculate the size after convolutions and pooling layers
        self.flattened_size = self._get_flattened_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Binary classification (Healthy/Unhealthy)

    def _get_flattened_size(self, input_size):
        """Calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 1, input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

