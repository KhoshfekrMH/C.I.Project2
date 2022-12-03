import main1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

globalDataFrame = main1.dfGlobal
X = globalDataFrame.drop(['label'], axis= 1)
Y = globalDataFrame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)
