import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

import consts
from utils import time_now


data = pd.read_csv(consts.TRAIN_PATH, delimiter=',', index_col='Id')
x = data.iloc[:, 0:-1]
y = data.SalePrice.values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=consts.DATA_TEST_SIZE, random_state=consts.RANDOM_STATE)

if __name__ == '__main__':
  print(f'{time_now()} Загрузка данных')
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=consts.DATA_TEST_SIZE, random_state=consts.RANDOM_STATE)
  
  print(f'{time_now()} Сохранение данных')
  Path(consts.X_TRAIN).mkdir(parents=True, exist_ok=True)
  with open(consts.X_TRAIN_FULL, 'wb') as f:
    pickle.dump(X_train, f)

  Path(consts.X_TEST).mkdir(parents=True, exist_ok=True)
  with open(consts.X_TEST_FULL, 'wb') as f:
    pickle.dump(X_test, f)

  Path(consts.Y_TRAIN).mkdir(parents=True, exist_ok=True)
  with open(consts.Y_TRAIN_FULL, 'wb') as f:
    pickle.dump(y_train, f)

  Path(consts.Y_TEST).mkdir(parents=True, exist_ok=True)
  with open(consts.Y_TEST_FULL, 'wb') as f:
    pickle.dump(y_test, f)