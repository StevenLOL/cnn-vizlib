import numpy as np
import os

from helpers import DataSet

DIE_X_FILE = os.path.join(os.path.dirname(__file__), 'die.X.npy')
DIE_Y_FILE = os.path.join(os.path.dirname(__file__), 'die.y.npy')

def die(standardization_type='individual'):
    X = np.load(DIE_X_FILE)
    # File is numbered from 1 to 6 (inclusive), and as strings
    y = np.load(DIE_Y_FILE).astype(np.int32) - 1
    return DataSet(X, y, label_lookup=dict(zip(y, y + 1))).standardize(standardization_type).shuffle()
