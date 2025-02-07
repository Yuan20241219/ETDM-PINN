import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


loss_1=pd.read_csv('result_data/train_loss_1.csv').iloc[:,1]
loss_10=pd.read_csv('result_data/train_loss_10.csv').iloc[:,1]
loss_40=pd.read_csv('result_data/train_loss_40.csv').iloc[:,1]
loss_1_nom=pd.read_csv('result_data/train_loss_1_nom.csv').iloc[:,1]

plt.figure(figsize=(10,5))
plt.plot(loss_1,label='n_features_1')
plt.plot(loss_10,label='n_features_10')
plt.plot(loss_40,label='n_features_40')
plt.legend(loc='best')
plt.ylabel('train_MSE')
plt.xlabel('epochs')
plt.show()


plt.figure(figsize=(10,5))
plt.plot(loss_1,label='未标准化')
plt.plot(loss_1_nom,label='标准化后')
plt.legend(loc='best')
plt.ylabel('train_MSE')
plt.xlabel('epochs')
plt.show()