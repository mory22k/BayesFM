import numpy as np

def remove_duplicate_rows(X, y, match_y=True):
    unique_rows, unique_indices, counts = np.unique(X, axis=0, return_index=True, return_counts=True)

    duplicate_indices = np.where(counts > 1)[0]
    if len(duplicate_indices) == 0:
        return X, y

    mask = np.ones(len(X), dtype=bool)
    for index in duplicate_indices:
        rows = np.where((X == unique_rows[index]).all(axis=1))[0]

        if match_y:
            unique_y, y_indices = np.unique(y[rows], return_index=True)
            mask[rows[np.setdiff1d(np.arange(len(rows)), y_indices)]] = False
        else:
            mask[rows[1:]] = False

    X_unique = X[mask]
    y_unique = y[mask]

    return X_unique, y_unique
