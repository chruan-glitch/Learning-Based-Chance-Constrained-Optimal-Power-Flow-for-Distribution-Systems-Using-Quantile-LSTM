import torch
import torch.nn as nn


class QRLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2, n_quantiles=3):
        super(QRLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size * n_quantiles)
        self.output_size = output_size
        self.n_quantiles = n_quantiles

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.view(-1, self.output_size, self.n_quantiles)


QUANTILES = [0.10, 0.50, 0.90]

def quantile_loss(preds, target, quantiles=QUANTILES):
    """preds: [batch, targets, qs],  target: [batch, targets]"""
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors).mean())
    return sum(losses)
