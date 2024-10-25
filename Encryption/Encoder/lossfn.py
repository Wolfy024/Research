import torch
from torch import nn

class MaxVarianceLossWithPenalty(nn.Module):
    def __init__(self):
        super(MaxVarianceLossWithPenalty, self).__init__()

    def forward(self, output):
        # Compute the variance across all dimensions
        variance = output.var(dim=0).sum()  # Sum variance across all dimensions
        variance_loss = -variance  # Maximize variance by minimizing negative variance

        # Initialize the penalty for repeated values
        h = 0

        # Flatten the output for easier comparison
        flattened_output = output.view(-1)

        # Count how many times each unique value appears
        unique_vals, counts = torch.unique(flattened_output, return_counts=True)

        # Penalize for repeated values
        for count in counts:
            if count > 1:
                h += count.item() - 1  # Add penalty for every repeated instance

        # Multiply the penalty by 10 as per the requirement
        penalty = h * 10

        # Combine the variance loss and the penalty
        total_loss = variance_loss + penalty

        return total_loss

