from torch import nn
import torch


class MeanVarianceLoss(nn.Module):
    """Mean-variance loss (Pan et al., CVPR 2018).

    Targets are class indices (age - min_age), so the mean and variance are computed over class
    indices as well; the age offset cancels in both terms.
    """

    def __init__(self, lambda_1, lambda_2):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, input, target):
        target = target.float()
        p = torch.softmax(input, dim=1)

        # mean loss
        a = torch.arange(input.shape[1], dtype=torch.float32, device=input.device)
        mean = (p * a).sum(1)
        mean_loss = ((mean - target) ** 2).mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1).mean()

        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss
