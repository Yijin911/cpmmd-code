"""Synthetic data generation for two-sample testing experiments."""

import numpy as np


def generate_data(n_samples, dist_type='gaussian', shift_type='none', delta=0.0,
                  dim=2):
    """Generate paired synthetic datasets for two-sample testing.

    Parameters
    ----------
    n_samples : int
        Number of samples per dataset.
    dist_type : str
        Base distribution: 'gaussian', 't_dist', or 'lognormal'.
    shift_type : str
        Type of shift applied to Y: 'none', 'mean', or 'variance'.
    delta : float
        Magnitude of the shift.
    dim : int
        Dimensionality of each sample (default 2).

    Returns
    -------
    X, Y : ndarray of shape (n_samples, dim)
    """

    if dist_type == 'gaussian':
        X = np.random.standard_normal((n_samples, dim))
        Y = np.random.standard_normal((n_samples, dim))
    elif dist_type == 't_dist':
        X = np.random.standard_t(df=3, size=(n_samples, dim))
        Y = np.random.standard_t(df=3, size=(n_samples, dim))
    elif dist_type == 'lognormal':
        X = np.random.lognormal(mean=0.0, sigma=1.0, size=(n_samples, dim))
        Y = np.random.lognormal(mean=0.0, sigma=1.0, size=(n_samples, dim))
    else:
        raise ValueError(f"Unknown dist_type: {dist_type!r}")

    if shift_type == 'none':
        pass
    elif shift_type == 'mean':
        Y = Y + delta
    elif shift_type == 'variance':
        Y = Y * (1.0 + delta)
    else:
        raise ValueError(f"Unknown shift_type: {shift_type!r}")

    return X, Y
