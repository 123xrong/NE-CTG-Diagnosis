import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        
        # Define the convolutional and pooling layers as sequences
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Compute the size of the flattened layer dynamically
        self.flattened_size = self._get_flattened_size(input_size)
        
        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

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
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self, input_size):
        super(EnhancedCNN, self).__init__()
        
        # Define the convolutional and pooling layers as sequences
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Compute the size of the flattened layer dynamically
        self.flattened_size = self._get_flattened_size(input_size)
        
        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_flattened_size(self, input_size):
        """Helper function to calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 2, input_size)
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x):
        x = self.features(x)  # Process through all feature layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)  # Process through all classifier layers
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes=2, lstm_hidden_size=64, lstm_layers=1):
        super(CNN_LSTM, self).__init__()

        # Kernel, stride, and padding setup
        kernel_size = 5  # To capture features with a receptive field size about 30 data points
        stride = 1
        padding = kernel_size // 2  # Padding to maintain dimensionality

        # Define the convolutional layers using Sequential
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Compute the size of the flattened layer dynamically for LSTM input
        self.flattened_size = self._get_flattened_size(input_size)
        
        # LSTM Layer setup
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_flattened_size(self, input_size):
        """Calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 1, input_size)  # Simulate input in the expected shape
        dummy_output = self.features(dummy_input)
        # Return the number of features per time-step to the LSTM
        print(dummy_output.shape)
        return dummy_output.shape[2]  # width dimension after conv layers

    def forward(self, x):
        x = self.features(x)
        #print(f"Shape after CNN: {x.shape}")
        #x = x.transpose(1, 2)  # Reshape for LSTM; needs (batch, seq_len, features)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take only the last LSTM output for classification

        # Fully connected layers
        x = self.classifier(x)
        return x

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()

        # Convolutional blocks using Sequential
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Dynamically compute the flattened size
        self.flattened_size = self._get_flattened_size(input_size)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_flattened_size(self, input_size):
        """Calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 1, input_size)
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x):
        # Process input through convolutional layers
        x = self.features(x)

        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)

        # Process input through the classifier layers
        x = self.classifier(x)
        return x
    
class EnhancedCNN1D(nn.Module):
    def __init__(self, input_size):
        super(EnhancedCNN1D, self).__init__()
        
        # Define the convolutional layers using Sequential
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Dynamically compute the flattened size for the fully connected layers
        self.flattened_size = self._get_flattened_size(input_size)

        # Fully connected layers using Sequential
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_flattened_size(self, input_size):
        """Helper function to calculate the flattened size after conv and pooling layers."""
        dummy_input = torch.randn(1, 1, input_size)  # Simulate input in the expected shape
        dummy_output = self.features(dummy_input)    # Use the convolutional layers
        return dummy_output.numel()

    def forward(self, x):
        # Process input through convolutional layers
        x = self.features(x)

        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)

        # Process input through the classifier layers
        x = self.classifier(x)
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=2, d_model=64, seq_len=1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Embedding layer to project input features to model dimensions
        self.embedding = nn.Linear(input_size, d_model)
    
        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(d_model * seq_len, 2)  # Direct linear transformation for classification

    def forward(self, x):
        # x shape: (batch_size, feature_num, seq_len)
        print(f"Input shape: {x.shape}")  # Debugging line to check input shape
        x = x.permute(0, 2, 1)  # Permute to (batch_size=32, seq_len, feature_num=d_model)
        x = self.embedding(x)  # Embedding to project features (batch_size, seq_len, d_model), (32, 7200, 64)
        x = x.permute(1, 0, 2)  # Permute to (seq_len, batch_size, d_model) for transformer (7200, 32, 64)

        x = self.transformer(x)  # Pass through transformer

        x = x.permute(1, 0, 2)  # Permute back to (batch_size, seq_len, d_model) (32, 7200, 64)
        x = x.flatten(start_dim=1)  # Flatten sequence steps and d_model dimensions (32, 7200*64)
        x = self.classifier(x)  # Apply classifier to generate final output (32, 2)
        return x

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=4, scaling_factor=0.01):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Rank of the approximation

        # LoRA layers will be added to the original weight matrix
        self.lora_A = nn.Parameter(torch.Tensor(self.original_layer.out_features, r))
        self.lora_B = nn.Parameter(torch.Tensor(r, self.original_layer.in_features))

        # Initialize parameters
        self._reset_parameters(scaling_factor)

    def _reset_parameters(self, scaling_factor):
        # Initialize LoRA matrices with scaled Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Scale initialization as per original layer
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))  # Scale initialization as per original layer

        # Scale down the initialization by a factor to manage training stability
        self.lora_A.data *= scaling_factor
        self.lora_B.data *= scaling_factor

    def forward(self, x):
        # Apply the original layer
        original_output = self.original_layer(x)

        # Calculate the low-rank update
        lora_update = F.linear(x, self.lora_A @ self.lora_B, bias=None)

        # Return the sum of the original output and the low-rank update
        return original_output + lora_update

