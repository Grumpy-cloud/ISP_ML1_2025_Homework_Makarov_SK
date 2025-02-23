import numpy as np


# 2 points
def euclidean_distance(X, Y) -> np.ndarray:
    """
    Compute element wise euclidean distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the Euclidean distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_sq = np.sum(X**2, axis=1)
    Y_sq = np.sum(Y**2, axis=1)
    XY = X @ Y.T
    X_sq = X_sq.reshape(-1, 1)
    return np.sqrt(X_sq - 2*XY + Y_sq)


# 2 points
def cosine_distance(X, Y) -> np.ndarray:
    """
    Compute element wise cosine distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the cosine distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_norm = np.sqrt(np.sum(X**2, axis=1))
    Y_norm = np.sqrt(np.sum(Y**2, axis=1))
    XY = X @ Y.T
    X_norm = X_norm.reshape(-1, 1)
    return 1 - XY / (X_norm * Y_norm)


# 1 point
def manhattan_distance(X, Y) -> np.ndarray:
    """
    Compute element wise manhattan distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the manhattan distance between the corresponding pair of vectors from the arrays X and Y
    """
    return np.abs(X[:,np.newaxis] - Y).sum(-1)