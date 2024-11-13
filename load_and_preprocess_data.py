import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import random

def segment_data(X, y, window_size=200, zero_ratio_threshold=0.05):
    """
    Segments each subject's data into windows of size 'window_size',
    discarding segments where more than 'zero_ratio_threshold' of the values are zero.
    
    Args:
        X (numpy array): The input data (shape: subjects, features, time_steps).
        y (numpy array): The labels (shape: subjects).
        window_size (int): The size of the window for segmentation.
        zero_ratio_threshold (float): Threshold for the maximum allowed ratio of zeros in a segment.
        
    Returns:
        segmented_X, segmented_y: Segmented data and labels.
    """
    segmented_X = []
    segmented_y = []

    # Iterate over each sample's data and segment it
    for i in range(X.shape[0]):
        sample_data = X[i]  # Shape: (time_steps,)
        sample_label = y[i]  # Single label for the sample
        
        # Segment the sample data
        num_segments = sample_data.shape[0] // window_size  # Total segments we can create
        for j in range(num_segments):
            start = j * window_size
            end = start + window_size
            segment = sample_data[start:end]  # Shape: (window_size,)
            
            # Count the number of zeros in the segment
            zero_count = np.sum(segment == 0)
            zero_ratio = zero_count / len(segment)
            
            # Discard segments with more than 'zero_ratio_threshold' zeros
            if zero_ratio > zero_ratio_threshold:
                continue
            
            # Append the valid segment and the corresponding label
            segmented_X.append(segment)
            segmented_y.append(sample_label)
    
    return np.array(segmented_X), np.array(segmented_y)


def load_and_preprocess_data(data_path, features_to_use=None, test_size=0.2, random_state=42, window_size=200, use_segmentation=False):
    """
    Loads, preprocesses, and optionally segments the data based on the provided features and window size.
    
    Args:
        data_path (str): Path to the data file (pickle or npz).
        features_to_use (list or None): A list specifying which features to use (e.g., ['FHR1', 'TOCO']).
                                        If None, defaults to using all available features.
        test_size (float): Proportion of the data to use for testing.
        window_size (int): The size of the window for segmentation.
        use_segmentation (bool): Whether to segment the data or not.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple of training and test tensors: (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    """
    
    # Load data
    if data_path.endswith('.pickle'):
        with open(data_path, 'rb') as f:
            X, y = pickle.load(f)
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
        X, y = data['X'], data['y']
    else:
        raise ValueError("Unsupported file type. Use a .pickle or .npz file.")
    
    # Select features based on the feature indices provided (i.e., features_to_use)
    if features_to_use and X.ndim == 3:
        # Assume feature names map to specific rows, e.g., 'FHR1' => X[:, 0, :], 'TOCO' => X[:, 1, :]
        feature_indices = {
            'FHR1': 0,
            'TOCO': 1,
            # Add more features here in the future, e.g., 'Feature3': 2, etc.
        }
        
        selected_indices = [feature_indices[feature] for feature in features_to_use]
        X_selected = X[:, selected_indices, :]  # Select only the features specified in features_to_use
    else:
        X_selected = X  # Use all features if none are specified
    
    if use_segmentation:
        # Segment the data into windows of size 'window_size'
        segmented_X, segmented_y = segment_data(X_selected, y, window_size=window_size)
        
        # Separate minority and majority classes
        minority_class = 1 if np.sum(segmented_y == 1) < np.sum(segmented_y == 0) else 0
        majority_class = 1 - minority_class
        
        # Get all segments of the minority class
        minority_X = segmented_X[segmented_y == minority_class]
        minority_y = segmented_y[segmented_y == minority_class]
        
        # Randomly sample the same number of segments from the majority class
        majority_X = segmented_X[segmented_y == majority_class]
        majority_y = segmented_y[segmented_y == majority_class]
        
        # Randomly sample the number of the length of the semented minority class from the majority class
        sampled_majority_indices = random.sample(range(len(majority_X)), len(minority_X))
        majority_X_sampled = majority_X[sampled_majority_indices]
        majority_y_sampled = majority_y[sampled_majority_indices]
        
        # Combine the minority class and the sampled majority class
        balanced_X = np.vstack((minority_X, majority_X_sampled))
        balanced_y = np.hstack((minority_y, majority_y_sampled))
        
        # Shuffle the balanced data
        shuffle_indices = np.random.permutation(len(balanced_y))
        balanced_X = balanced_X[shuffle_indices]
        balanced_y = balanced_y[shuffle_indices]
        
        # Train-test split (balanced data)
        X_train, X_test, y_train, y_test = train_test_split(balanced_X, balanced_y, test_size=test_size, random_state=random_state, stratify=balanced_y)
        print(f"Data dimension after segmentation: {X_train.shape} (training), {X_test.shape} (testing)")

    else:
        # No segmentation: use original data, but apply class balancing
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Normalize each feature (handle both 2D and 3D data)
    if X_train.ndim == 3:
        # For 3D data (multiple features like FHR1 and TOCO)
        scalers = [StandardScaler() for _ in range(X_train.shape[1])]  # One scaler per feature
        X_train_norm = np.stack([scaler.fit_transform(X_train[:, i, :]) for i, scaler in enumerate(scalers)], axis=1)
        X_test_norm = np.stack([scaler.transform(X_test[:, i, :]) for i, scaler in enumerate(scalers)], axis=1)
    else:
        # For 2D data (only one feature like FHR1)
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)  # Normalize the 2D data
        X_test_norm = scaler.transform(X_test)
    
    # Convert to tensors (reshape if necessary for 2D case)
    if X_train_norm.ndim == 2:
        X_train_tensor = torch.tensor(X_train_norm[:, np.newaxis, :], dtype=torch.float32)  # Add a channel dimension for 1D CNN
        X_test_tensor = torch.tensor(X_test_norm[:, np.newaxis, :], dtype=torch.float32)
    else:
        X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

