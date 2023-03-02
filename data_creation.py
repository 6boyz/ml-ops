import pickle
import consts
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

def true_fun(x, a=np.pi, b = 0, f=np.sin):
    x = np.atleast_1d(x)[:]
    a = np.atleast_1d(a)
    
    if f is None: f = lambda x:x # line
    x = np.sum([ai*np.power(x, i+1) for i,ai in enumerate(a)],axis=0)

    return f(x+ b)

def noises(shape , noise_power):
    return np.random.randn(*shape) *noise_power

def dataset(a, b, f = None,  N = 250, x_max =1, noise_power = 0, random_x = True,  seed = 42):
    np.random.seed(seed)
    
    if random_x:
        x = np.sort(np.random.rand(N))*x_max    
    else:
        x = np.linspace(0,x_max,N)
    
    y_true = np.array([])
    
    for f_ in np.append([], f):
        y_true=np.append(y_true, true_fun(x, a, b, f_))
    
    y_true = y_true.reshape(-1,N).T
    y = y_true + noises(y_true.shape , noise_power)

    return y, y_true, np.atleast_2d(x).T

def time_now():
   return datetime.now().strftime("%H:%M:%S")

if __name__ == '__main__':
  print(f'{time_now()} Генерация данных')
  y, _, x = dataset(a = [1,4,-2], b = 1, f = consts.DATA_FUNCTION, N = consts.DATA_N, x_max = 1, noise_power = consts.DATA_NOISE_POWER, seed = 42)
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