import numpy as np
import torch

def get_wind_power_smooth(wind, window=5):
    smoothed = np.convolve(wind, np.ones(window)/window, mode='same')
    return smoothed

def load_and_preprocess_all(n_samples=1000, input_dim=10):
    # 👉 用你 notebook 的真实数据替换这里
    X = np.random.randn(n_samples, input_dim)
    y = np.sum(X, axis=1, keepdims=True)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y
