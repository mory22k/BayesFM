from bayesfm.utils import remove_duplicate_rows
import numpy as np

def test_remove_duplicate_rows():
    X = np.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
        [3, 4]
    ])
    y = np.array([10, 10, 20, 10, 20, 20])

    X_unique, y_unique = remove_duplicate_rows(X, y, match_y=True)
    y_unique_true = np.array([10, 20, 10, 20])
    print("Unique X (match_y=True):\n", X_unique)
    print("Unique y (match_y=True):\n", y_unique)
    assert np.all(y_unique == y_unique_true)

    X_unique, y_unique = remove_duplicate_rows(X, y, match_y=False)
    y_unique_true = np.array([10, 10])
    print("Unique X (match_y=False):\n", X_unique)
    print("Unique y (match_y=False):\n", y_unique)
    assert np.all(y_unique == y_unique_true)

if __name__ == '__main__':
    test_remove_duplicate_rows()
