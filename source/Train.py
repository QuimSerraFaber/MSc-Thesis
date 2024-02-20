import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

def training_single_model(data, model, loss_function, batch_size=256, lr=0.001, patience=5, epochs=50, progress=False):
    """
    Trains a single model to predict all four parameters.

    Parameters:
    data (dict): The data dictionary containing the noisy TAC signals and ground truth parameters.
    model (nn.Module): The neural network model to train.
    loss_function (nn.Module): The loss function to use.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for the optimizer.
    patience (int): The patience for early stopping.
    epochs (int): The maximum number of epochs to train.
    progress (bool): Whether to print the validation loss at each epoch.

    Returns:
    nn.Module: The trained model.
    float: The best validation loss.
    np.ndarray: The mean percentage difference for each parameter.
    np.ndarray: The standard deviation of the percentage difference for each parameter.
    """
    # Extract the data from the dictionary
    inputs = data["noisy_tacs"]
    true_params = data["gt_parameters"]

    # Convert the arrays to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    true_params_tensor = torch.tensor(true_params[:, :-1], dtype=torch.float32) # Remove the last column (k4)

    # Split data into training and validation sets
    total_samples = inputs_tensor.shape[0]
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs_tensor, true_params_tensor)

    # Randomly split dataset into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for both training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    in_features = inputs_tensor.shape[1]
    model = model(in_features)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_loss = np.inf  # Initialize the best validation loss as infinity
    patience_counter = 0  # Counter for epochs waited
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for inputs, true_params in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_params = model(inputs)

            # Compute the loss
            loss = loss_function(predicted_params, true_params)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the loss for monitoring
            total_loss += loss.item()
        
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, true_params in val_dataloader:
                predicted_params = model(inputs)
                loss = loss_function(predicted_params, true_params)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        if progress == True:
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:  # Check for improvement with a minimum delta
            best_val_loss = avg_val_loss  # Update the best validation loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increase patience counter
        
        if patience_counter >= patience:  # Check if patience counter has reached the patience limit
            print(f"Stopping early at epoch {epoch + 1}")
            break

    # Final evaluation on the validation set
    true_params_list = []
    predicted_params_list = []

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, true_params in val_dataloader:
            predicted_params = model(inputs)
            loss = loss_function(predicted_params, true_params)
            total_val_loss += loss.item()
            true_params_list.append(true_params.numpy())
            predicted_params_list.append(predicted_params.numpy())

    # Concatenate all batches to get the whole validation set predictions
    true_params_concat = np.concatenate(true_params_list, axis=0)
    predicted_params_concat = np.concatenate(predicted_params_list, axis=0)

    # Compute the percentile differences for each parameter
    abs_diff = np.abs(true_params_concat - predicted_params_concat)
    epsilon = 1e-8  # Small constant to avoid division by zero
    percentage_diff = (abs_diff / (true_params_concat + epsilon)) * 100

    # Calculate the mean and standard deviation of the percentage differences
    mean_percentage_diff = np.mean(percentage_diff, axis=0)
    std_percentage_diff = np.std(percentage_diff, axis=0)
    print("Mean percentage difference:", mean_percentage_diff)
    print("Standard deviation of percentage difference:", std_percentage_diff)
    
    return model, best_val_loss, mean_percentage_diff, std_percentage_diff


