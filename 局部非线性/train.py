
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
from structure import Structure
from load_iter import SeqDataIter
from ETDM_matric import ETDM_structure
from E_PINN import E_pinn
from torch import nn


def V2Fnl(V):
    V_re = V[:, 0, 0, :]
    Fnl = 1000 * K[0] * (V_re ** 3)
    return Fnl


if __name__ == '__main__':
    dof = 5
    M = [5000] * 5
    K = [1e6] * 5
    zeta = 0.05
    dt = 0.02
    n_steps = 1500
    lr = 0.1
    epochs = 300
    batch_size =100
    n_features =10
    result_idx = [0]


    train_data = pd.read_csv("../data/地震加速度.csv").iloc[1:201, 1:]


    L = torch.zeros((dof, 2))
    L[:, 0] = torch.tensor(M)
    L[:, 1] = -torch.tensor(([1, 0, 0, 0, 0]))

    structure = Structure(dof, M, K, num_omega1=1, num_omega2=5, zeta=zeta)
    structure = ETDM_structure(L, structure, dt=dt, n_steps=n_steps)
    load_iter = SeqDataIter(train_data, batch_size, n_features)
    e_pinn_model = E_pinn(n_features=n_features, A=structure.A, lr=lr)

    timestamp1 = time.time()
    for name, param in e_pinn_model.lstm_model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    train_loss = e_pinn_model.train(load_iter, epochs, result_idx,V2Fnl)
    timestamp2 = time.time()
    time_difference = timestamp2 - timestamp1
    print("时间差（秒）：", time_difference)


    plt.figure(figsize=(15, 5))
    plt.plot(range(epochs), train_loss, label="训练损失")
    plt.xlabel("epoches")
    plt.ylabel("MSEloss")
    plt.legend(loc='best')
    plt.show()

    df = pd.DataFrame(train_loss)
    df.to_csv("result_data/train_loss_10.csv")
    torch.save(e_pinn_model, "E_pinn_model.pt")