from sklearn.metrics import r2_score

import consts
from utils import load_data

model = load_data(consts.MODEL_FULL)
x_test = load_data(consts.X_TEST_FULL)
y_test = load_data(consts.Y_TEST_FULL)

y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print('Coefficient of determination score: ', score)