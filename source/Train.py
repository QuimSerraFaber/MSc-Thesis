import torch
import torch.optim as optim
import torch.optim.lr_scheduler
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
    config (dict): A dictionary containing the configuration values for training the model.
    data (dict): The data dictionary containing the noisy TAC signals and ground truth parameters.
    model_class (nn.Module): The neural network model to train.
    loss_function (nn.Module): The loss function to use.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for the optimizer.
    patience (int): The patience for early stopping.
    epochs (int): The maximum number of epochs to train.
    progress (bool): Whether to print the validation loss at each epoch.
    TAC_loss (bool): Whether to use the TAC loss or traditional loss.

    Returns:
    nn.Module: The trained model.
    dict: A dictionary containing the best validation loss, mean percentage difference, and standard deviation of the percentage difference.
    """
    # Extracting configuration values
    data = config['data']
    model_class = config['model_class']
    loss_function = config['loss_function']
    batch_size = config.get('batch_size', 1024)
    lr = config.get('lr', 0.001)
    patience = config.get('patience', 5)
    epochs = config.get('epochs', 50)
    progress = config.get('progress', False)
    TAC_loss = config.get('TAC_loss', False)  # Whether to use the TAC loss or traditional loss
    fast = config.get('fast', True)  # Whether to use only the 22 closest indices to the measuring points

    # Extract the data from the data dictionary
    inputs = data["noisy_tacs"]
    true_params = data["gt_parameters"]

    # Use only the 22 closest indices to the measuring points
    if fast == True:
        closest_indices = [0, 4, 6, 8, 10, 14, 18, 31, 54, 77, 100, 140, 196, 282, 396, 567, 794, 1022, 1250, 1478, 1705, 1933]
        inputs = inputs[:, closest_indices]
        num_equidistant_points = len(closest_indices)
    else:
        num_equidistant_points = 2048

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

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
                loss = loss_function(predicted_params, inputs, num_equidistant_points) 
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
                    loss = loss_function(predicted_params, inputs, num_equidistant_points)
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
        
        if patience_counter >= patience and epoch > 10:  # Check if patience counter has reached the patience limit
            if progress == True:
                print(f"Stopping early at epoch {epoch + 1}")
            break

        # Step the scheduler
        if epoch < 101:
            scheduler.step()

    # Final evaluation on the validation set
    true_params_list = []
    predicted_params_list = []

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, true_params in val_dataloader:
            predicted_params = model(inputs)
            if TAC_loss:
                loss = loss_function(predicted_params, inputs, num_equidistant_points)
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

    # Create dictionary with all the results
    results = {
        "true_params": true_params_concat,
        "predicted_params": predicted_params_concat
    }
    
    return model, results


def training_parallel_models(config):
    """
    Trains a group of models parallelly for each parameter.

    Parameters:
    config (dict): The configuration dictionary containing the training parameters.
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
    # Unpack the configuration dictionary
    data = config['data']
    model_class = config['model_class']
    loss_function = config['loss_function']
    batch_size = config.get('batch_size', 1024)
    lr = config.get('lr', 0.001)
    patience = config.get('patience', 5)
    epochs = config.get('epochs', 50)
    progress = config.get('progress', False)
    TAC_loss = config.get('TAC_loss', False)  # Whether to use the TAC loss or traditional loss
    fast = config.get('fast', True)  # Whether to use only the 22 closest indices to the measuring points

    # Extract the data from the dictionary
    inputs = data["noisy_tacs"]
    true_params = data["gt_parameters"]

    # Use only the 22 closest indices to the measuring points
    if fast == True:
        closest_indices = [0, 4, 6, 8, 10, 14, 18, 31, 54, 77, 100, 140, 196, 282, 396, 567, 794, 1022, 1250, 1478, 1705, 1933]
        inputs = inputs[:, closest_indices]
        num_equidistant_points = len(closest_indices)
    else:
        num_equidistant_points = 2048

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
    
    # Add scheduler to each optimizer
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8) for optimizer in optimizers]

    # Early stopping
    best_val_loss = np.inf  # Initialize the best validation loss as infinity
    patience_counter = 0  # Counter for epochs waited

    # Training loop
    for epoch in range(epochs):
        # Set models to training mode
        for model in models:
            model.train()
            
        # Iterate over training data
        for inputs, true_params in train_dataloader:
            current_predictions = []
            
            # Collect predictions from all models without computing the loss yet
            for i, model in enumerate(models):
                predicted_param = model(inputs).squeeze()
                current_predictions.append(predicted_param.unsqueeze(1))  # Make it a column vector
                
            # Concatenate predictions along dimension 1 to form a [batch_size, 4] tensor
            all_predictions = torch.cat(current_predictions, dim=1)
            
            # Compute the loss
            if TAC_loss:
                # Note: Assuming the TAC loss function is designed to take all predictions and inputs
                loss = loss_function(all_predictions, inputs, num_equidistant_points)
            else:
                # This branch may not be used, but included for completeness
                loss = loss_function(all_predictions, true_params)
            
            # Backward pass and optimization
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            loss.backward()
            
            for optimizer in optimizers:
                optimizer.step()

            
        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, true_params in val_dataloader:
                predicted_params_list = []

                for i, model in enumerate(models):
                    # Forward pass
                    predicted_param = model(inputs).squeeze()
                    predicted_params_list.append(predicted_param.unsqueeze(1))
                
                # Concatenate the predictions along dimension 1
                predicted_params = torch.cat(predicted_params_list, dim=1)

                # Compute loss 
                if TAC_loss:
                    loss = loss_function(predicted_params, inputs, num_equidistant_points)
                else:
                    loss = loss_function(predicted_params, true_params)

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
        
        if patience_counter >= patience and epoch > 10:  # Check if patience counter has reached the patience limit
            if progress == True:
                print(f"Stopping early at epoch {epoch + 1}")
            break

        # Step the scheduler
        if epoch < 101:
            for scheduler in schedulers:
                scheduler.step()
    
    # Final evaluation on the validation set
    # Initialize lists to accumulate the true and predicted parameters
    accumulated_true_params = []
    accumulated_predicted_params = []

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, true_params in val_dataloader:
            predicted_eval_params_list = []

            for i, model in enumerate(models):
                # Forward pass
                predicted_param = model(inputs).squeeze()
                predicted_eval_params_list.append(predicted_param.unsqueeze(1))
            
            # Concatenate the predictions along dimension 1
            predicted_eval_params = torch.cat(predicted_eval_params_list, dim=1)

            # Compute the loss
            if TAC_loss:
                loss = loss_function(predicted_eval_params, inputs, num_equidistant_points)
            else:
                loss = loss_function(predicted_eval_params, true_params)

            # Accumulate the loss
            total_val_loss += loss.item()

            # Accumulate the true and predicted parameters for later analysis
            accumulated_true_params.append(true_params.numpy())
            accumulated_predicted_params.append(predicted_eval_params.numpy())
    
    # After accumulating results from all batches, concatenate them
    true_params_concat = np.concatenate(accumulated_true_params, axis=0)
    predicted_params_concat = np.concatenate(accumulated_predicted_params, axis=0)

    # Compute k_i for the true and predicted parameters
    true_ki = ki_macro(true_params_concat[:, 0], true_params_concat[:, 1], true_params_concat[:, 2])
    predicted_ki = ki_macro(predicted_params_concat[:, 0], predicted_params_concat[:, 1], predicted_params_concat[:, 2])

    # Append k_i as a new column to the true and predicted params
    true_params_concat = np.column_stack((true_params_concat, true_ki))
    predicted_params_concat = np.column_stack((predicted_params_concat, predicted_ki))

    # Create dictionary with all the results
    results = {
        "true_params": true_params_concat,
        "predicted_params": predicted_params_concat
    }
    
    return models, results

