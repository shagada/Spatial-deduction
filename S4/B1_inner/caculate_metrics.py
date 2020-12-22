import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
def MAPE(X,y):
    return np.mean(np.abs((X-y)/y))
def RMSE(X,y):
    return np.sqrt(np.sum(np.mean(np.square(X-y))))
def MAE(X,y):
    return np.mean(np.abs(X-y))
true_file=pd.read_csv('B1_inner.csv')
pre_file=pd.read_csv('B1_inner_p1.csv',header=None)
test=[]
for k in range(436):
    p=[]
    a1 = pre_file.iloc[3 * k, :].values
    a2 = pre_file.iloc[3 * k + 1, :].values
    a3 = pre_file.iloc[3 * k + 2, :].values
    test.append(a1[46])
print(RMSE(np.array(test),true_file.iloc[:,21].values))
print(MAE(np.array(test),true_file.iloc[:,21].values))
print(MAPE(np.array(test),true_file.iloc[:,21].values))
print(pearsonr(np.array(test),true_file.iloc[:,21].values)[0])

plt.plot([i for i in range(436)],test,color='r')
plt.plot([i for i in range(436)],true_file.iloc[:,21].values)
plt.show()

