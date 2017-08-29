import numpy as np
import pandas as pd
import random
from logistic_regression import my_logistic_regression

"""
Evaluation Steps

1. Divide df_competition_evaluation into training and test called df_training, df_test
2. Run your "my_logistic_regression" function on df_training, retrieving a0, a1, a2, a3
3. Evaluate the accuracy of a0, a1, a2, a3 by calculating the AUC on df_testing
"""

# read data
df_competition_evaluation = pd.read_csv('df_competition.csv')
n = len(df_competition_evaluation)

# divide into training and test
l = range(0, n)
k = int(0.6 * (n))
training_index = random.sample(l, k)
df_training = df_competition_evaluation.iloc[training_index]
test_index = [x for x in l if x not in training_index]
df_test = df_competition_evaluation.iloc[test_index]
glm_results = my_logistic_regression(df_training)

# calculate the raw predictions on the test set
a0 = glm_results[0]
a1 = glm_results[1]
a2 = glm_results[2]
a3 = glm_results[3]

glm_pred = 1 / \
    (1 + np.exp(-(a0 + a1 * df_test['V1'] +
     a2 * df_test['V2'] + a3 * df_test['V3'])))

# calculate the ROC
TPR = []
FPR = []
thresholds = np.arange(0, 1.001, 0.001)
m = len(thresholds)
pd.options.mode.chained_assignment = None  # default='warn'
df_test['glm_pred'] = glm_pred

for threshold in thresholds:
    TP = len(df_test[(df_test['Target'] == 1) &
                     (df_test['glm_pred'] > threshold)])
    FN = len(df_test[(df_test['Target'] == 1) &
                     (df_test['glm_pred'] < threshold)])
    FP = len(df_test[(df_test['Target'] == 0) &
                     (df_test['glm_pred'] > threshold)])
    TN = len(df_test[(df_test['Target'] == 0) &
                     (df_test['glm_pred'] < threshold)])
    TPR.append(TP / (TP + FN))
    FPR.append(FP / (FP + TN))

TPR[0] = 1
FPR[0] = 1
TPR[m - 1] = 0
FPR[m - 1] = 0

# integrating the ROC to get the AUC
c = np.array([sum(n) / 2 for n in zip(*[TPR[0:(m - 1)], TPR[1:m]])])
AUC = abs(round(sum(np.diff(FPR) * c), 3))
print(AUC)
