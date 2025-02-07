
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
from structure import Structure
from load_iter import SeqDataIter
from ETDM_matric import ETDM_structure
import torch.nn as nn


def V2Fnl(V):
    V_re = V[:, 0, 0, :]
    Fnl = 100 * K[0] * (V_re ** 3)
    return Fnl

def train_fine_tuning(net=None, learning_rate=None):

    params_1x = [param for name, param in net.named_parameters()
         if name not in ["fc.weight", "fc.bias","fc1.weight", "fc1.bias"]]
    trainer = torch.optim.Adam([{'params': params_1x},
                               {'params': net.fc1.parameters(),
                                'lr': learning_rate*10},
                                {'params': net.fc.parameters(),
                                'lr': learning_rate*10}],
                            lr=learning_rate, weight_decay=0.998)

    return trainer



if __name__ == '__main__':
    dof = 5
    M = [1000] * 5
    K = [1e5] * 5
    zeta = 0.05
    dt = 0.02
    n_steps = 1500
    lr = 1e-3
    epochs = 10
    batch_size =50
    n_features =10
    result_idx = [0]

    fine_tune_train= pd.read_csv("../../data/地震加速度.csv").iloc[200:250, 1:]
    fine_tune_test = pd.read_csv("../../data/地震加速度.csv").iloc[251, 1:]


    L = torch.zeros((dof, 2))
    L[:, 0] = torch.tensor(M)
    L[:, 1] = -torch.tensor(([1, 0, 0, 0, 0]))

    structure = Structure(dof, M, K, num_omega1=1, num_omega2=5, zeta=zeta)
    structure = ETDM_structure(L, structure, dt=dt, n_steps=n_steps)
    load_iter = SeqDataIter(fine_tune_train, batch_size, n_features)

    e_pinn_model = torch.load("模型参数/E_pinn_model.pt")
    # nn.init.xavier_uniform_(e_pinn_model.lstm_model.fc.weight)

    e_pinn_model.lstm_model.to("cuda")
    e_pinn_model.A=structure.A
    before_result=e_pinn_model.predict(fine_tune_test)

    timestamp1 = time.time()
    e_pinn_model.optimizer = train_fine_tuning(e_pinn_model.lstm_model, learning_rate=lr)
    train_loss = e_pinn_model.train(load_iter, epochs, result_idx,V2Fnl)
    after_result=e_pinn_model.predict(fine_tune_test)
    timestamp2 = time.time()
    time_difference = timestamp2 - timestamp1
    print("时间差（秒）：", time_difference)

# 打印训练损失值
    plt.figure(figsize=(15, 5))
    plt.plot(range(epochs), train_loss, label="训练损失")
    plt.xlabel("epoches")
    plt.ylabel("MSEloss")
    plt.legend(loc='best')
    plt.show()

# 打印微调前后的结果
    true_result=pd.read_csv("../result_data/Newmark_ns251_2.csv").iloc[0,1:]
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(true_result)),np.array(true_result),label="NewMark")  #真实值
    plt.plot(range(before_result.shape[-1]),-before_result[0,0,0,:].detach(),label="before") #微调前
    plt.plot(range(after_result.shape[-1]),-after_result[0,0,0,:].detach(),label="after") #微调后
    plt.legend(loc='best')
    plt.xlabel("时间步")
    plt.ylabel('位移/m')
    plt.show()

# 打印微调前后的误差
    plt.figure(figsize=(15, 5))
    plt.plot(range(before_result.shape[-1]),-before_result[0,0,0,:].detach()-np.array(true_result),label="before") #微调前
    plt.plot(range(after_result.shape[-1]),-after_result[0,0,0,:].detach()-np.array(true_result),label="after") #微调后
    plt.legend(loc='best')
    plt.xlabel("时间步")
    plt.ylabel('位移/m')
    plt.show()

