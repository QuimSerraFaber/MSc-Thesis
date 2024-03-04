import numpy as np
import torch
import torch.nn as nn
from Data_simulation import IRF, equidistant_interpolation, c_tissue, simulated_tac


def compute_parameter_loss(predicted_param, true_param, use_absolute=False):
    # Loss is the difference between the predicted and true parameter
    # Input must be torch tensors
    if use_absolute:
        # Compute the absolute difference
        return (predicted_param - true_param).abs().mean()
    else:
        # Compute the regular mean difference
        return (predicted_param - true_param).mean()
    

def TAC_loss(true_param, predicted_param, num_equidistant_points = 2048):
    """
    Calculate the loss between the true and predicted TAC values.

    Parameters:
    true_param (torch.Tensor): True parameter values
    predicted_param (torch.Tensor): Predicted parameter values
    num_equidistant_points (int): Number of equidistant points to interpolate the plasma and blood concentration values

    Returns:
    torch.Tensor: Mean squared error between the true and predicted TAC values
    """
    
    rtim_list = [0.125, 0.291667, 0.375, 0.458333, 0.583333, 0.75, 0.916667, 1.5, 2.5, 3.5, 4.5, 6.25, 8.75, 12.5, 
                 17.5, 25, 35, 45, 55, 65, 75, 85]
    pl_list = [0.05901, 0.0550587, 110.023, 83.0705, 55.6943, 44.4686, 36.9873, 27.5891, 13.5464, 6.33916, 3.52664, 
               2.49758, 1.44494, 1.04103, 0.71615, 0.52742, 0.43791, 0.35239, 0.32866, 0.27326, 0.26068, 0.24129]
    bl_list = [0.08737, 0.081723, 164.763, 125.181, 84.5654, 68.4428, 58.1711, 44.4919, 24.3138, 14.0272, 9.47834, 
               7.88443, 5.59859, 4.81384, 3.79608, 3.04436, 2.67881, 2.23747, 2.13455, 1.80559, 1.74821, 1.63998]

    # Interpolate plasma and blood concentration values
    new_rtim, _, _, pchip_pl = equidistant_interpolation(rtim_list, pl_list, num_equidistant_points)
    _, _, _, pchip_bl = equidistant_interpolation(rtim_list, bl_list, num_equidistant_points)

    # Convert parameter tensors to numpy arrays
    true_param = true_param.detach().cpu().numpy()
    predicted_param = predicted_param.detach().cpu().numpy()

    # Calculate the impulse response functions:
    true_irf = IRF(true_param, new_rtim)
    pred_irf = IRF(predicted_param, new_rtim)

    # Calculate the C-Tissue values
    dt = new_rtim[1] - new_rtim[0]
    true_ct = c_tissue(true_irf, pchip_pl, dt)
    pred_ct = c_tissue(pred_irf, pchip_pl, dt)

    # Calculate the TAC values
    true_tac = simulated_tac(true_ct, true_param, pchip_bl)
    pred_tac = simulated_tac(pred_ct, predicted_param, pchip_bl)

    # Calculate the mean squared error between the true and predicted TAC
    return nn.MSELoss()(torch.tensor(true_tac), torch.tensor(pred_tac))




