from S9 import caculate_similarity as cs
import pandas as pd
import tensorflow as tf
import numpy as np
import warnings
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import csv
warnings.filterwarnings('ignore')
def MAPE(X,y):
    return np.mean(np.abs((X-y)/y))
def MAPE1(X,y):
    return np.mean(np.abs((y-X)/y))
def RMSE(X,y):
    return np.sqrt(np.sum(np.mean(np.square(X-y))))
def MAE(X,y):
    return np.mean(np.abs(X-y))
def get_data(filename,index):
    file=pd.read_csv(filename)
    input_data=file.iloc[index,:6].values
    val_data=file.iloc[index,6:7].values
    test_data=file.iloc[index,7]
    return input_data*200,val_data*200,test_data*200
fd1,fd2,fd3=cs.caculate()
X=tf.placeholder(tf.float32,[1,6])
U=tf.Variable(tf.random_normal([3,2]))
V=tf.Variable(tf.random_normal([2,50]))
x=tf.matmul(U,V)


pred1=x[0:1,3:4]
pred2=x[0:1,12:13]

pred3=x[1:2,12:13]


pred4=x[2:3,32:33]
pred5=x[2:3,3:4]
pred6=x[2:3,12:13]



pred=tf.concat([pred1,pred2,pred3,pred4,pred5,pred6],1)
loss=tf.reduce_mean(tf.square(tf.subtract(pred,X)))
for i in range(3):
    if i == 0:
        for j in range(49):
            loss+=0.1/fd1[j]*tf.reduce_mean(tf.square(tf.subtract(x[i:i + 1, j:j + 1], x[i:i + 1, j + 1:j + 2])))

    elif i == 1:
        for j in range(49):
            loss+=0.1/fd2[j]*tf.reduce_mean(tf.square(tf.subtract(x[i:i + 1, j:j + 1], x[i:i + 1, j + 1:j + 2])))
    else:
        for j in range(49):
            loss+=0.1/fd3[j]*tf.reduce_mean(tf.square(tf.subtract(x[i:i + 1, j:j + 1], x[i:i + 1, j + 1:j + 2])))
for i in range(3):
    for j in range(50):
        loss+=0.1*tf.reduce_mean(tf.square(tf.subtract(x[i:i+1,j:j+1],x[i:i+1,49-j:49-j+1])))
opt=tf.train.GradientDescentOptimizer(learning_rate=0.00004).minimize(loss)
epoches=10000
with tf.Session() as sess:
    for k in range(436):
        val_loss=[]
        print(k)
        input_data,val_data, test_data = get_data('B7_middle.csv', k)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            p,_,cost=sess.run([x,opt,loss],feed_dict={
                X:[input_data]
            })
            if epoch%50==0:
                val_=np.array([p[2][18]])
                val_loss.append(RMSE(val_,val_data))
                if len(val_loss)>25 and val_loss[-1]>val_loss[-2] and val_loss[-2]>val_loss[-3]:
                    break
        result=sess.run(x,feed_dict={
            X:[input_data]
        })
        # print(result)
        file2 = open('B7_middle_p1.csv', 'a+', newline='')
        writer1 = csv.writer(file2)
        for row in result:
            writer1.writerow(row/200)