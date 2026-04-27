"""
src/model.py
QR-LSTM quantile regression model + pinball loss
"""

import torch
import torch.nn as nn

QUANTILES = [0.10, 0.50, 0.90]


class QRLSTM(nn.Module):
    """
    Two-layer LSTM with a linear head that predicts multiple quantiles simultaneously.

    Args:
        input_size  : Number of input features
        hidden_size : LSTM hidden dimension (default 128)
        output_size : Number of prediction targets (default 2: Solar + Wind)
        n_quantiles : Number of quantiles (default 3: q10 / q50 / q90)
    """

    def __init__(self, input_size, hidden_size=128, output_size=2, n_quantiles=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size * n_quantiles)
        self.output_size = output_size
        self.n_quantiles = n_quantiles

    def forward(self, x):
        """
        Args:
            x   : [batch, seq_len, input_size]
        Returns:
            out : [batch, output_size, n_quantiles]
        """
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.view(-1, self.output_size, self.n_quantiles)


def quantile_loss(preds, target, quantiles=QUANTILES):
    """
    Pinball loss (quantile regression loss).

    Args:
        preds     : [batch, output_size, n_quantiles]
        target    : [batch, output_size]
        quantiles : list of float, e.g. [0.10, 0.50, 0.90]

    Returns:
        Scalar loss (mean over batch, targets, and quantiles)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors).mean())
    return sum(losses)
