import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from Train import training_single_model, training_parallel_models
from models.Initial_fc_nn import FC_single
from models.FC_nn_single_bounded import FC_single_bounded
from models.FC_nn_parallel import *
from Losses import TAC_loss

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the data and define the loss function
#loss = nn.MSELoss()
#loss = nn.L1Loss()
loss = TAC_loss
config = { 
    'data': np.load("data/Generated_Data/simulation_simple_0.01.npz"),
    'model_class': FC_single_bounded,
    'loss_function': loss,
    'batch_size': 1028,
    'lr': 0.001,
    'patience': 5,
    'epochs': 50,
    'progress': True,
    'TAC_loss': True
}

# Initialize lists to collect the arrays
mean_percentage_diffs = []
std_percentage_diffs = []
mean_diffs = []
std_diffs = []
n_models = 5
for i in range(n_models):
    print(f"Training model {i + 1}")
    model, results = training_single_model(config)
    # Append the results to the lists
    mean_percentage_diffs.append(results["mean_percentage_diff"])
    std_percentage_diffs.append(results["std_percentage_diff"])
    mean_diffs.append(results["mean_diff"])
    std_diffs.append(results["std_diff"])

# Convert lists to 2D numpy arrays
mean_percentage_diffs_array = np.array(mean_percentage_diffs)
std_percentage_diffs_array = np.array(std_percentage_diffs)
mean_diffs_array = np.array(mean_diffs)
std_diffs_array = np.array(std_diffs)

# Calculate the average of each column
mean_percentage_diffs_avg = np.mean(mean_percentage_diffs_array, axis=0)
std_percentage_diffs_avg = np.mean(std_percentage_diffs_array, axis=0)
mean_diffs_avg = np.mean(mean_diffs_array, axis=0)
std_diffs_avg = np.mean(std_diffs_array, axis=0)

# Print the average values for each column
print("Average of mean percentage differences:", mean_percentage_diffs_avg)
print("Average of std percentage differences:", std_percentage_diffs_avg)
print("Average of mean differences:", mean_diffs_avg)
print("Average of std differences:", std_diffs_avg)

# Custom parameter labels
parameters = ['k1', 'k2', 'k3', 'vb', 'ki']

# Plotting the percentile differences
plt.figure(figsize=(10, 6))
# Error bars for the standard deviation (symmetrical)
errorbar_container = plt.errorbar(parameters, mean_percentage_diffs_avg, 
             yerr=[std_percentage_diffs_avg, std_percentage_diffs_avg],  # Symmetrical error bars
             fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5, 
             linestyle='None', label='Average Std')

# Annotating each point with its mean value +- std
for i, (mean, std) in enumerate(zip(mean_percentage_diffs_avg, std_percentage_diffs_avg)):
    plt.annotate(f'{mean:.2f} ± {std:.2f}', (parameters[i], mean), textcoords="offset points", xytext=(10,0), ha='left')


# Extending the graph to the right by adjusting x-axis limits
plt.xlim(-0.25, len(parameters)-0.1)  # Set dynamic limits based on the number of parameters

# Customizing the plot
plt.title(f'Percentual difference in predictions: {model_class.__name__}, {loss} & {n_models} models')
plt.xlabel('Parameter')
plt.ylabel('Average Percentual Difference [%]')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.axhline(0, color='black', linestyle='--')  # Dashed line at 0

# Handling legend for both Average Difference and Average Std
plt.scatter(parameters, mean_percentage_diffs_avg, color='blue', label='Average Difference')
plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Average Percentual Difference'),
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Average Std of Percentual Difference')],
           loc='best')

# Show plot
plt.show()

# Plotting the absolute differences
plt.figure(figsize=(10, 6))
# Error bars for the standard deviation
errorbar_container = plt.errorbar(parameters, mean_diffs_avg,
                yerr=[std_diffs_avg, std_diffs_avg],  # Symmetrical error bars
                fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5,
                linestyle='None', label='Average Std')

# Annotating each point with its value
for i, (mean, std) in enumerate(zip(mean_diffs_avg, std_diffs_avg)):
    plt.annotate(f'{mean:.2f} ± {std:.2f}', (parameters[i], mean), textcoords="offset points", xytext=(10,0), ha='left')

# Extending the graph to the right by adjusting x-axis limits
plt.xlim(-0.25, len(parameters)-0.1)  # Set dynamic limits based on the number of parameters

# Customizing the plot
plt.title(f'Difference in predictions: {model_class.__name__}, {loss} & {n_models} models')
plt.xlabel('Parameter')
plt.ylabel('Average Absolute Difference')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.axhline(0, color='black', linestyle='--')  # Dashed line at 0

# Handling legend for both Average Difference and Average Std
plt.scatter(parameters, mean_diffs_avg, color='blue', label='Average Difference')
plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Average Difference'),
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Average Std of Difference')],
           loc='best')

# Show plot
plt.show()