import numpy as np

RANDOM_STATE = 42

DATA_TEST_SIZE = .3
DATA_N = 10_000
DATA_NOISE_POWER = .1
DATA_FUNCTION = lambda x: np.power(x, 2) - 5 * x

Y_TRAIN = '.\\train\\'
Y_TRAIN_FULL = Y_TRAIN + 'y.pckl'
X_TRAIN = '.\\train\\'
X_TRAIN_FULL = X_TRAIN + 'x.pckl'
Y_TEST = '.\\test\\'
Y_TEST_FULL = Y_TEST + 'y.pckl'
X_TEST = '.\\test\\'
X_TEST_FULL = X_TEST + 'x.pckl'

DATA_PATH = '.\\data_default\\' + 'train.csv'
