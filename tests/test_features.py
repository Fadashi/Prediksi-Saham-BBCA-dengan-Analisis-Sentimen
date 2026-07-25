"""Unit test untuk windowing sequence dan feature assembly."""

import numpy as np
from src.features.assemble import create_sliding_window_sequences


def test_create_sliding_window_sequences():
    features = np.arange(100).reshape(50, 2)
    targets = np.arange(50)
    lookback = 10

    X, y = create_sliding_window_sequences(features, targets, lookback=lookback)

    assert X.shape == (40, 10, 2)
    assert y.shape == (40,)
    assert y[0] == targets[10]
