import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Train import training_single_model, training_parallel_models
from models.Initial_fc_nn import FC_single
from models.FC_nn_single_bounded import FC_single_bounded
from models.FC_nn_parallel import FC_parallel
from Losses import TAC_loss

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the data and define the loss function
data = np.load("data/Generated_Data/simulation_simple_0.01.npz")
model_class = FC_parallel
#loss = nn.MSELoss()
#loss = nn.L1Loss()
loss = TAC_loss

# Initialize lists to collect the arrays
mean_percentage_diffs = []
std_percentage_diffs = []
mean_absolute_diffs = []
std_absolute_diffs = []
for i in range(10):
    print(f"Training model {i + 1}")
    model, results = training_parallel_models(data, model_class, loss, batch_size=1028, lr=0.001, patience=5, epochs=50, progress=True)
    # Append the results to the lists
    mean_percentage_diffs.append(results["mean_percentage_diff"])
    std_percentage_diffs.append(results["std_percentage_diff"])
    mean_absolute_diffs.append(results["mean_abs_diff"])
    std_absolute_diffs.append(results["std_abs_diff"])

# Convert lists to 2D numpy arrays
mean_percentage_diffs_array = np.array(mean_percentage_diffs)
std_percentage_diffs_array = np.array(std_percentage_diffs)
mean_absolute_diffs_array = np.array(mean_absolute_diffs)
std_absolute_diffs_array = np.array(std_absolute_diffs)

# Calculate the average of each column
mean_percentage_diffs_avg = np.mean(mean_percentage_diffs_array, axis=0)
std_percentage_diffs_avg = np.mean(std_percentage_diffs_array, axis=0)
mean_absolute_diffs_avg = np.mean(mean_absolute_diffs_array, axis=0)
std_absolute_diffs_avg = np.mean(std_absolute_diffs_array, axis=0)

# Print the average values for each column
print("Average of mean percentage differences:", mean_percentage_diffs_avg)
print("Average of std percentage differences:", std_percentage_diffs_avg)
print("Average of mean absolute differences:", mean_absolute_diffs_avg)
print("Average of std absolute differences:", std_absolute_diffs_avg)

# Custom parameter labels
parameters = ['k1', 'k2', 'k3', 'vb', 'ki']

# Plotting the percentile differences
plt.figure(figsize=(10, 6))
# Error bars for the standard deviation
errorbar_container = plt.errorbar(parameters, mean_percentage_diffs_avg, 
             yerr=[np.zeros_like(std_percentage_diffs_avg), std_percentage_diffs_avg], 
             fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5, 
             linestyle='None', label='Average Std')

# Annotating each point with its value
for i, txt in enumerate(mean_percentage_diffs_avg):
    plt.annotate(f'{txt:.2f}', (parameters[i], mean_percentage_diffs_avg[i]), textcoords="offset points", xytext=(25,0), ha='center')
for i, txt in enumerate(std_percentage_diffs_avg):
    plt.annotate(f'{txt:.2f}', (parameters[i], mean_percentage_diffs_avg[i] + std_percentage_diffs_avg[i]), textcoords="offset points", xytext=(25,0), ha='center')

# Extending the graph to the right by adjusting x-axis limits
plt.xlim(-0.25, len(parameters)-0.5)  # Set dynamic limits based on the number of parameters

# Customizing the plot
plt.title('Percentile difference in predictions for a fully connected NN: MSE Loss & 10 models')
plt.xlabel('Parameter')
plt.ylabel('Mean Percentage Difference [%]')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Handling legend for both Average Difference and Average Std
plt.scatter(parameters, mean_percentage_diffs_avg, color='blue', label='Average Difference')
plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Mean Percentage Difference'),
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Mean Std of Percentage Difference')],
           loc='best')

# Show plot
plt.show()

# Plotting the absolute differences
plt.figure(figsize=(10, 6))
# Error bars for the standard deviation
errorbar_container = plt.errorbar(parameters, mean_absolute_diffs_avg, 
             yerr=[np.zeros_like(std_absolute_diffs_avg), std_absolute_diffs_avg], 
             fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5, 
             linestyle='None', label='Average Std')

# Annotating each point with its value
for i, txt in enumerate(mean_absolute_diffs_avg):
    plt.annotate(f'{txt:.2f}', (parameters[i], mean_absolute_diffs_avg[i]), textcoords="offset points", xytext=(25,0), ha='center')
for i, txt in enumerate(std_absolute_diffs_avg):
    plt.annotate(f'{txt:.2f}', (parameters[i], mean_absolute_diffs_avg[i] + std_absolute_diffs_avg[i]), textcoords="offset points", xytext=(25,0), ha='center')

# Extending the graph to the right by adjusting x-axis limits
plt.xlim(-0.25, len(parameters)-0.5)  # Set dynamic limits based on the number of parameters

# Customizing the plot
plt.title('Asbolute difference in predictions for a fully connected NN: MSE Loss & 10 models')
plt.xlabel('Parameter')
plt.ylabel('Mean Absolute Difference')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Handling legend for both Average Difference and Average Std
plt.scatter(parameters, mean_absolute_diffs_avg, color='blue', label='Average Difference')
plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Mean Absolute Difference'),
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Mean Std of Absolute Difference')],
           loc='best')

# Show plot
plt.show()