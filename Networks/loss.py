import numpy as np
import torch

class NegativeLogLikelihoodSurvivalLoss:
    def __call__(self, hazards, S, Y, c, alpha=0.15, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)
        c = c.view(batch_size, 1).float()
        S_padded = torch.cat([torch.ones_like(c), S], 1)
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
            torch.gather(hazards, 1, Y).clamp(min=eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss
