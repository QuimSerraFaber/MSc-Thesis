import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

def ki_macro(k1, k2, k3):
    """
    Computes the macro parameter k_i from the parameters k1, k2, and k3.

    Parameters:
    k1 (float): The first parameter.
    k2 (float): The second parameter.
    k3 (float): The third parameter.

    Returns:
    float: The macro rate constant k_i.
    """
    ki = (k1 * k2) / (k2 + k3)
    return ki

def training_single_model(config):
    """
    Trains a single model to predict all four parameters.

    Parameters:
    config (dict): A dictionary containing all the settings for training the model.

    Returns:
    nn.Module: The trained model.
    dict: A dictionary containing the best validation loss, mean percentage difference, and standard deviation of the percentage difference.
    """
    # Extracting configuration values
    data = config['data']
    model_class = config['model_class']
    loss_function = config['loss_function']
    batch_size = config.get('batch_size', 1028)  # Example of providing default values
    lr = config.get('lr', 0.001)
    patience = config.get('patience', 5)
    epochs = config.get('epochs', 50)
    progress = config.get('progress', False)
    TAC_loss = config.get('TAC_loss', False)  # Whether to use the TAC loss or traditional loss


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
    model = model_class(in_features)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_loss = np.inf  # Initialize the best validation loss as infinity
    patience_counter = 0  # Counter for epochs waited
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        for inputs, true_params in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_params = model(inputs)

            # Compute the loss
            if TAC_loss:
                loss = loss_function(predicted_params, inputs)
            else:
                loss = loss_function(predicted_params, true_params)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
        
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, true_params in val_dataloader:
                predicted_params = model(inputs)
                if TAC_loss:
                    loss = loss_function(predicted_params, inputs)
                else:
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
            if progress == True:
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
            if TAC_loss:
                loss = loss_function(predicted_params, inputs)
            else:
                loss = loss_function(predicted_params, true_params)
            total_val_loss += loss.item()
            true_params_list.append(true_params.numpy())
            predicted_params_list.append(predicted_params.numpy())

    # Concatenate all batches to get the whole validation set predictions
    true_params_concat = np.concatenate(true_params_list, axis=0)
    predicted_params_concat = np.concatenate(predicted_params_list, axis=0)

    # Compute k_i for the true and predicted parameters
    true_ki = ki_macro(true_params_concat[:, 0], true_params_concat[:, 1], true_params_concat[:, 2])
    predicted_ki = ki_macro(predicted_params_concat[:, 0], predicted_params_concat[:, 1], predicted_params_concat[:, 2])

    # Append k_i as a new column to the true and predicted params
    true_params_concat = np.column_stack((true_params_concat, true_ki))
    predicted_params_concat = np.column_stack((predicted_params_concat, predicted_ki))

    # Compute the percentile differences for each parameter
    diff = true_params_concat - predicted_params_concat
    epsilon = 1e-8  # Small constant to avoid division by zero
    percentage_diff = (diff / (true_params_concat + epsilon)) * 100

    # Calculate the mean and standard deviation of the percentage differences
    mean_percentage_diff = np.mean(percentage_diff, axis=0)
    std_percentage_diff = np.std(percentage_diff, axis=0)
    if progress == True:
        print("Mean percentage difference:", mean_percentage_diff)
        print("Standard deviation of percentage difference:", std_percentage_diff)

    # Calculate the mean and standard deviation of the differences
    mean_diff = np.mean(diff, axis=0)
    std_diff = np.std(diff, axis=0)
    if progress == True:
        print("Mean difference:", mean_diff)
        print("Standard deviation of difference:", std_diff)

    # Create dictionary with all the results
    results = {
        "best_val_loss": best_val_loss,
        "mean_percentage_diff": mean_percentage_diff,
        "std_percentage_diff": std_percentage_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff
    }
    
    return model, results


def training_parallel_models(data, model_class, loss_function, batch_size=256, lr=0.001, patience=5, epochs=50, progress=False):
    """
    Trains a group of models parallelly for each parameter.

    Parameters:
    data (dict): The data dictionary containing the noisy TAC signals and ground truth parameters.
    model_class (nn.Module): The neural network model to train.
    loss_function (nn.Module): The loss function to use.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for the optimizer.
    patience (int): The patience for early stopping.
    epochs (int): The maximum number of epochs to train.
    progress (bool): Whether to print the validation loss at each epoch.

    Returns:
    nn.Module: The trained model.
    dict: A dictionary containing the best validation loss, mean percentage difference, and standard deviation of the percentage difference.
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

    # Initialize the models for each parameter
    models = [model_class(in_features=inputs_tensor.shape[1]) for _ in range(4)]

    # Optimizers (one for each model)
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]

    # Early stopping
    best_val_loss = np.inf  # Initialize the best validation loss as infinity
    patience_counter = 0  # Counter for epochs waited

    # Training loop
    for epoch in range(epochs):
        for inputs, true_params in train_dataloader:
            for i, model in enumerate(models):
                # Set the model to training mode
                model.train()

                # Get the optimizer for the current model
                optimizer = optimizers[i]
                optimizer.zero_grad()

                # Forward pass
                predicted_param = model(inputs).squeeze()
                
                # Compute loss for the specific parameter
                loss = loss_function(predicted_param, true_params[:, i])

                # Backward pass and optimization
                loss.backward()

                # Update the weights
                optimizer.step()
            
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, true_params in val_dataloader:
                for i, model in enumerate(models):
                    # Forward pass
                    predicted_param = model(inputs).squeeze()

                    # Compute loss for the specific parameter
                    loss = loss_function(predicted_param, true_params[:, i])

                    # Accumulate the loss
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
            if progress == True:
                print(f"Stopping early at epoch {epoch + 1}")
            break
    
    # Final evaluation on the validation set
    # Initialize lists for true parameters and predictions for each model
    true_params_lists = [[] for _ in range(len(models))]
    predicted_params_lists = [[] for _ in range(len(models))]

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, true_params in val_dataloader:
            for i, model in enumerate(models):
                # Forward pass
                predicted_param = model(inputs).squeeze()

                # Compute loss for the specific parameter
                loss = loss_function(predicted_param, true_params[:, i])

                # Accumulate the loss
                total_val_loss += loss.item()

                # Append the true and predicted parameters to their respective lists
                true_params_lists[i].append(true_params[:, i].numpy())
                predicted_params_lists[i].append(predicted_param.numpy())
    
    # Concatenate all batches for each parameter to get the whole validation set predictions
    true_params_concat = np.concatenate([np.concatenate(lst, axis=0).reshape(-1, 1) for lst in true_params_lists], axis=1)
    predicted_params_concat = np.concatenate([np.concatenate(lst, axis=0).reshape(-1, 1) for lst in predicted_params_lists], axis=1)

    # Compute k_i for the true and predicted parameters
    true_ki = ki_macro(true_params_concat[:, 0], true_params_concat[:, 1], true_params_concat[:, 2])
    predicted_ki = ki_macro(predicted_params_concat[:, 0], predicted_params_concat[:, 1], predicted_params_concat[:, 2])

    # Append k_i as a new column to the true and predicted params
    true_params_concat = np.column_stack((true_params_concat, true_ki))
    predicted_params_concat = np.column_stack((predicted_params_concat, predicted_ki))

    # Compute the percentile differences for each parameter
    diff = true_params_concat - predicted_params_concat
    epsilon = 1e-8  # Small constant to avoid division by zero
    percentage_diff = (diff / (true_params_concat + epsilon)) * 100

    # Calculate the mean and standard deviation of the percentage differences
    mean_percentage_diff = np.mean(percentage_diff, axis=0)
    std_percentage_diff = np.std(percentage_diff, axis=0)
    if progress == True:
        print("Mean percentage difference:", mean_percentage_diff)
        print("Standard deviation of percentage difference:", std_percentage_diff)

    # Calculate the mean and standard deviation of the absolute differences
    mean_diff = np.mean(diff, axis=0)
    std_diff = np.std(diff, axis=0)
    if progress == True:
        print("Mean absolute difference:", mean_diff)
        print("Standard deviation of absolute difference:", std_diff)
    
    # Create dictionary with all the results
    results = {
        "best_val_loss": best_val_loss,
        "mean_percentage_diff": mean_percentage_diff,
        "std_percentage_diff": std_percentage_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff
    }
    
    return model, results

