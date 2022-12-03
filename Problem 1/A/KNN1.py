from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import main1
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# region KNN with Accuracy
globalDataFrame = main1.dfGlobal
X = globalDataFrame.drop(['label'], axis= 1)
Y = globalDataFrame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)

def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual)) * 100
    return mape

KNN_Model = KNeighborsRegressor(n_neighbors=3).fit(X_train,Y_train)

KNN_predict = KNN_Model.predict(X_test)

# Accuracy
KNN_MAPE = MAPE(Y_test,KNN_predict)
Accuracy_KNN = 100 - KNN_MAPE
print("MAPE: ",KNN_MAPE)
print('Accuracy of KNN model: {:0.2f}%.'.format(Accuracy_KNN))
print("MAPE means Mean Absolute Percentage Error :)")
# endregion
