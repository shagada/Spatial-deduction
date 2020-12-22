from S4 import caculate_similarity as cs
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
    input_data=file.iloc[index,:18].values
    val_data=file.iloc[index,18:21].values
    test_data=file.iloc[index,21]
    return input_data*200,val_data*200,test_data*200
fd1,fd2,fd3=cs.caculate()
X=tf.placeholder(tf.float32,[1,18])
U=tf.Variable(tf.random_normal([3,2]))
V=tf.Variable(tf.random_normal([2,50]))
x=tf.matmul(U,V)



pred2=x[0:1,1:2]
pred3=x[0:1,7:8]
pred4=x[0:1,22:23]
pred1=x[0:1,27:28]
pred5=x[0:1,36:37]
pred6=x[0:1,41:42]
pred7=x[0:1,32:33]


pred8=x[1:2,46:47]
pred21=x[1:2,1:2]
pred9=x[1:2,7:8]
pred10=x[1:2,22:23]
pred11=x[1:2,36:37]
pred12=x[1:2,41:42]
pred13=x[1:2,32:33]


pred14=x[2:3,46:47]
pred15=x[2:3,1:2]
pred16=x[2:3,7:8]
pred17=x[2:3,22:23]

pred=tf.concat([pred2,pred3,pred4,pred1,pred5,pred6,pred7,pred8,pred21,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17],1)
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
        input_data,val_data, test_data = get_data('B1_inner.csv', k)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            p,_,cost=sess.run([x,opt,loss],feed_dict={
                X:[input_data]
            })
            if epoch%50==0:
                val_=np.array([p[2][36],p[2][41],p[2][32]])
                val_loss.append(RMSE(val_,val_data))
                if len(val_loss)>25 and val_loss[-1]>val_loss[-2] and val_loss[-2]>val_loss[-3]:
                    break
        result=sess.run(x,feed_dict={
            X:[input_data]
        })
        file2 = open('B1_inner_p1.csv', 'a+', newline='')
        writer1 = csv.writer(file2)
        for row in result:
            writer1.writerow(row/200)