import numpy as np
import pandas as pd

# region create two group samples with Label
mean1 = np.array([2.5, .25])

cov1 = np.array([[1, .75],
                 [.75, 1]])

mean2 = np.array([-1.5, 0])

cov2 = np.array([[2, .5],
                 [.5, 2]])

sampleOfTraining1 = np.random.multivariate_normal(mean1, cov1, 70)
sampleOfTraining2 = np.random.multivariate_normal(mean2, cov2, 300)

dfSample1 = pd.DataFrame(sampleOfTraining1)
dfSample2 = pd.DataFrame(sampleOfTraining2)

label1 = []
label2 = []
for i in range(70):
    i = 1
    label1.append(i)

for m in range(300):
    m = 2
    label2.append(m)
# adding two dataFrame sample 1 and 2 to one global dataFrame
dfSample1['label'] = label1
dfSample2['label'] = label2

frames = [dfSample1, dfSample2]
dfGlobal = pd.concat(frames)
# endregion

