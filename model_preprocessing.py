import pickle
import pandas as pd
import consts
import re

x_train = pd.read_pickle(consts.X_TRAIN_FULL)
y_train = pd.read_pickle(consts.Y_TRAIN_FULL)

x_test = pd.read_pickle(consts.X_TEST_FULL)
y_test = pd.read_pickle(consts.Y_TEST_FULL)

y_train = pd.DataFrame(y_train)



#
