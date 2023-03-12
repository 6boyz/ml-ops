import pandas as pd
from sklearn.model_selection import train_test_split

import consts
from utils import print_log, save_data

print_log('Loading data')
data = pd.read_csv(consts.DATA_PATH, delimiter=',', index_col='Id')
x = data.iloc[:, 0:-1]
y = data.SalePrice.values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=consts.DATA_TEST_SIZE, random_state=consts.RANDOM_STATE)

print_log('Saving data')

save_data(X_train, consts.X_TRAIN, consts.X_TRAIN_FULL)
save_data(X_test, consts.X_TEST, consts.X_TEST_FULL)
save_data(y_train, consts.Y_TRAIN, consts.Y_TRAIN_FULL)
save_data(y_test, consts.Y_TEST, consts.Y_TEST_FULL)
