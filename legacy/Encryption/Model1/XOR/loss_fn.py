import torch


class EncryptionLoss(torch.nn.Module):
    def __init__(self, multiplier, mean_weight=1.0, variance_weight=1.0, std_weight=1.0):
        super(EncryptionLoss, self).__init__()
        self.multiplier = multiplier
        self.mean_weight = mean_weight
        self.variance_weight = variance_weight
        self.std_weight = std_weight

    def forward(self, data):
        # Mean Loss: Minimize |mean| to keep it close to zero
        mean_loss = self.mean_weight * torch.abs(data.mean())

        # Variance Loss: Maximize variance by making it negative
        variance_loss = -self.variance_weight * data.var()

        # Standard Deviation Loss: Maximize std deviation by making it negative
        std_loss = -self.std_weight * torch.std(data)

        # Total Loss: Weighted sum of mean, variance, and std losses
        total_loss = mean_loss + variance_loss + std_loss

        return total_loss
