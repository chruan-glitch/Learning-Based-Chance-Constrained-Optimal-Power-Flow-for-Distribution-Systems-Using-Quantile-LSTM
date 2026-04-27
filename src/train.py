"""
src/train.py
QRLSTM training loop with cosine annealing scheduler and best-weight checkpointing
"""

import torch
from src.model import quantile_loss


def train_model(
    model,
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    lr=5e-4,
    weight_decay=1e-5,
    t_max=150,
    eta_min=1e-5,
    verbose=True,
):
    """
    Train QRLSTM and return the best-weight model with loss history.

    Args:
        model        : QRLSTM instance
        X_train      : torch.FloatTensor  [N, seq_len, n_features]
        y_train      : torch.FloatTensor  [N, n_targets]
        epochs       : Number of training epochs
        batch_size   : Mini-batch size
        lr           : Adam initial learning rate
        weight_decay : L2 regularisation coefficient
        t_max        : CosineAnnealingLR period (epochs)
        eta_min      : Minimum learning rate
        verbose      : Print progress every 25 epochs

    Returns:
        model        : Model loaded with best weights, set to eval mode
        loss_history : list[float]  average pinball loss per epoch
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )

    n_train    = X_train.shape[0]
    best_loss  = float("inf")
    best_state = None
    loss_history = []

    if verbose:
        print(f"Training QRLSTM  ({epochs} epochs, hidden=128, 2-layer LSTM)")
        print(f"  Train samples: {n_train},  Features: {X_train.shape[2]}")
        print("-" * 50)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n_train)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            loss = quantile_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(
                f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  best={best_loss:.5f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"\nTraining complete. Best loss = {best_loss:.5f}")

    return model, loss_history


def evaluate(model, X_test, y_test, scaler, target_indices, num_samples=None):
    """
    Run inference on the test set and return inverse-transformed predictions.

    Args:
        model          : Trained QRLSTM in eval mode
        X_test         : torch.FloatTensor
        y_test         : torch.FloatTensor
        scaler         : Fitted MinMaxScaler
        target_indices : list[int]
        num_samples    : Use only the first N samples (None = all)

    Returns:
        dict with keys 'true', 'q10', 'q50', 'q90'
        Each value is ndarray of shape [N, n_targets] in MW
    """
    import numpy as np

    if num_samples is not None:
        X_test = X_test[:num_samples]
        y_test = y_test[:num_samples]

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()   # [N, targets, quantiles]
    y_true = y_test.numpy()              # [N, targets]

    def _inv(arr, t_idx):
        dummy = np.zeros((len(arr), len(scaler.scale_)))
        dummy[:, target_indices[t_idx]] = arr
        raw = scaler.inverse_transform(dummy)[:, target_indices[t_idx]] / 1000.0
        return np.maximum(raw, 0.0)

    n_targets = y_true.shape[1]
    result    = {"true": [], "q10": [], "q50": [], "q90": []}
    for t in range(n_targets):
        result["true"].append(_inv(y_true[:, t], t))
        result["q10"].append(_inv(y_pred[:, t, 0], t))
        result["q50"].append(_inv(y_pred[:, t, 1], t))
        result["q90"].append(_inv(y_pred[:, t, 2], t))

    return {k: np.stack(v, axis=1) for k, v in result.items()}
