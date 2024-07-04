import numpy as np
import time
from fmgibbs import fm

def test_fm_fast():
    x = np.random.uniform(-1, 1, 100)
    b = np.random.uniform(-1, 1)
    w = np.random.uniform(-1, 1, 100)
    v = np.random.uniform(-1, 1, (100, 10))

    time_1 = time.time()
    fm_a = fm.fm_naive(x, b, w, v)
    time_2 = time.time()
    fm_b = fm.fm_fast(x, b, w, v)
    time_3 = time.time()

    print(f'Naive FM: {time_2 - time_1:10.3e}')
    print(f'Fast FM:  {time_3 - time_2:10.3e}')

    assert np.isclose(fm_a, fm_b)

if __name__ == '__main__':
    test_fm_fast()
