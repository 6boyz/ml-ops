import consts
import pickle
from sklearn.metrics import r2_score

model = pickle.load(open(MODEL_FULL, "rb"))
x_test = pickle.load(open(X_TEST_FULL, "rb"))
y_test = pickle.load(open(Y_TEST_FULL, "rb"))

y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print('Coefficient of determination score: ', score)