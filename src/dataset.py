"""
src/dataset.py
Data loading and preprocessing: physical modelling + leak-free train/test split + sliding windows
"""

import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_wind_power_smooth(v, rho, p_rated=1000, v_in=3, v_r=12, v_out=25):
    """
    Wind power curve with air density correction and piecewise smooth transitions.

    Args:
        v       : Hub-height wind speed (m/s)  [array]
        rho     : Air density (kg/m³)          [array]
        p_rated : Rated power (W)
        v_in    : Cut-in wind speed (m/s)
        v_r     : Rated wind speed (m/s)
        v_out   : Cut-out wind speed (m/s)

    Returns:
        p       : Wind power (W)               [array]
    """
    p = np.zeros_like(v, dtype=float)

    mask_cubic = (v >= v_in) & (v < v_r)
    p[mask_cubic] = (
        p_rated
        * (v[mask_cubic] ** 3 - v_in ** 3)
        / (v_r ** 3 - v_in ** 3)
        * (rho[mask_cubic] / 1.225)
    )

    mask_rated = (v >= v_r) & (v < v_out)
    p[mask_rated] = p_rated

    return np.clip(p, 0, p_rated)


def load_and_preprocess_all(file_pattern, window_size=24, alpha=0.20, h_hub=80):
    """
    Load weather CSV files and produce leak-free train/test sliding windows.

    Pipeline:
        1. Load and concatenate all matching CSVs
        2. Physical feature engineering (wind power curve, solar efficiency correction)
        3. Cyclic temporal encoding (hour, month)
        4. Chronological 80/20 split (no future leakage)
        5. Fit MinMaxScaler on train only, transform both splits
        6. Build sliding windows of length `window_size`

    Args:
        file_pattern : Glob pattern, e.g. "data/900131_*.csv"
        window_size  : Look-back window length in hours
        alpha        : Wind shear exponent for hub-height extrapolation
        h_hub        : Hub height (m); reference height assumed 10 m

    Returns:
        X_train, X_test  : torch.FloatTensor  [N, window_size, n_features]
        y_train, y_test  : torch.FloatTensor  [N, 2]  (Solar, Wind)
        scaler           : Fitted MinMaxScaler (for inverse transform)
        target_indices   : list[int]  indices of target columns in feature array
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    df = pd.concat(
        [pd.read_csv(f, skiprows=2) for f in files], axis=0
    ).reset_index(drop=True)

    df = df.dropna(
        subset=["Wind Speed", "Temperature", "Pressure", "GHI"]
    ).reset_index(drop=True)

    # -- Wind power: hub-height extrapolation + air density correction
    R_gas       = 287.05
    temp_k      = df["Temperature"] + 273.15
    pressure_pa = df["Pressure"] * 100.0
    df["rho"]   = np.clip(pressure_pa / (R_gas * temp_k), 0.9, 1.3)
    df["v_hub"] = df["Wind Speed"] * (h_hub / 10.0) ** alpha
    df["Wind_Power"] = get_wind_power_smooth(df["v_hub"].values, df["rho"].values)

    # -- Solar power: GHI with temperature coefficient correction (-0.4 %/°C above 25°C)
    t_cell = df["Temperature"] + 0.03 * df["GHI"]
    df["Solar_Power"] = np.clip(
        (df["GHI"] / 1000.0) * (1 - 0.004 * (t_cell - 25)), 0, None
    )

    # -- Cyclic temporal encoding
    df["hr_sin"] = np.sin(2 * np.pi * df["Hour"]  / 24.0)
    df["hr_cos"] = np.cos(2 * np.pi * df["Hour"]  / 24.0)
    df["mo_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["mo_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    features_cols  = [
        "hr_sin", "hr_cos", "mo_sin", "mo_cos",
        "Temperature", "Pressure", "Solar_Power", "Wind_Power",
    ]
    target_cols    = ["Solar_Power", "Wind_Power"]
    target_indices = [features_cols.index(t) for t in target_cols]

    # -- Leak-free split: split first, then fit scaler on train only
    split_idx    = int(len(df) * 0.8)
    train_df     = df.iloc[:split_idx].copy()
    test_df      = df.iloc[split_idx:].copy()

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features_cols])
    test_scaled  = scaler.transform(test_df[features_cols])

    # -- Sliding windows
    def _create_windows(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window : i, :])
            y.append(data[i, target_indices])
        return np.array(X), np.array(y)

    X_train, y_train = _create_windows(train_scaled, window_size)
    X_test,  y_test  = _create_windows(test_scaled,  window_size)

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_train),
        torch.FloatTensor(y_test),
        scaler,
        target_indices,
    )
