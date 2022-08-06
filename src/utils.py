import pandas as pd
import numpy as np

def zScore(X, dim=0):
    """
    z-score normalize time series. This method works even when nan values are included

    Inputs
    ------
    X : a time series stored as a numpy array or pandas dataframe
        X is a time series, typically of shape (timesteps, variables).
    dim : int
        Denotes which variable is the time dimension. If the input series are of shape (timesteps, variables), dim should be 0.
        If the shape is (variables, timesteps) it should be 1. The same applies for more dimensions: If the same is (timesteps,
        participants, variables), for example, the dim should be 0.

    Returns
    -------
    This function returns a new time series of the same shape as the input, but for which the mean of each variable is 0 and
    its standard deviation is 1.
    """

    # --- if X is a pandas dataframe, change it to a numpy array ---
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.to_numpy()

    # --- subtract mean and divide by standard deviation ---
    # Add tiny value to denominator to avoid dividing by 0
    return (X - np.nanmean(X, dim, keepdims=True)) / (np.nanstd(X, dim, keepdims=True) + 1e-7)

def moving_average(X, w):
    """
    Smooth an input time series by replacing values with the average of their neighbors

    Inputs
    ------
    X : a time series stored as a numpy array
        X is a time series of shape (timesteps, variables).
    w : int
        The window size for timesteps surrounding each value to compute the average
    """
    return np.convolve(X, np.ones(w), 'valid') / w
