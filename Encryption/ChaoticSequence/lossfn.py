import torch
import torch.nn as nn


class CustomVarianceLossWithPenalty(nn.Module):
    def __init__(self):
        super(CustomVarianceLossWithPenalty, self).__init__()

    def forward(self, output, target):
        # Calculate MSE loss
        mse_loss = nn.MSELoss()(output, target)

        # Calculate penalty for repeated values
        flattened_output = output.view(-1)
        unique_vals, counts = torch.unique(flattened_output, return_counts=True)
        penalty = sum(count - 1 for count in counts if count > 1)
        # Combine MSE and penalty
        total_loss = mse_loss + 5 * penalty  # Adjust the penalty coefficient as needed
        if penalty < 10:
            total_loss = total_loss - 5 * penalty - 100
        return total_loss
