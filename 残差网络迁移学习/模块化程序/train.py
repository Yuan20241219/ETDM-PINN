
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


def V2Fnl(V):
    V_re = V[:, 0, 0, :]
    Fnl = 1000 * K[0] * (V_re ** 3)
    return Fnl

def train_fine_tuning(net=None, learning_rate=None):
    params_1x = [param for name, param in net.named_parameters()
         if name not in ["fc.weight", "fc.bias"]]
    trainer = torch.optim.Adam([{'params': params_1x},
                               {'params': net.fc.parameters(),
                                'lr': learning_rate }],
                            lr=learning_rate, weight_decay=0.99)

    return trainer


if __name__ == '__main__':
    dof = 5
    M = [5000] * 5
    K = [1e6] * 5
    zeta = 0.05
    dt = 0.02
    n_steps = 1500
    lr = 0.01
    epochs = 100
    batch_size =180
    n_features =10
    result_idx = [0]


    train_data = pd.read_csv("../../data/地震加速度.csv").iloc[1:200, 1:]
    vali_data=pd.read_csv("../../data/地震加速度.csv").iloc[200:250, 1:]


    L = torch.zeros((dof, 2))
    L[:, 0] = torch.tensor(M)
    L[:, 1] = -torch.tensor(([1, 0, 0, 0, 0]))

    structure = Structure(dof, M, K, num_omega1=1, num_omega2=5, zeta=zeta)
    structure = ETDM_structure(L, structure, dt=dt, n_steps=n_steps)
    load_iter = SeqDataIter(train_data, batch_size, n_features)
    e_pinn_model = E_pinn(n_features=n_features, A=structure.A, lr=lr)
    train_loss = e_pinn_model.train(load_iter, epochs, result_idx,V2Fnl)


    vali_data = pd.read_csv("../../data/地震加速度.csv").iloc[251, 1:]
    v=e_pinn_model.predict(vali_data)

    plt.figure(figsize=(15, 5))
    plt.plot(range(epochs), train_loss, label="训练损失")
    plt.xlabel("epoches")
    plt.ylabel("MSEloss")
    plt.legend(loc='best')
    plt.show()

    df = pd.DataFrame(train_loss)
    df.to_csv("result_data/get_V_loss_nom.csv")
    torch.save(e_pinn_model, "模型参数/E_pinn_model.pt")
    torch.save(e_pinn_model.lstm_model.state_dict(), '模型参数/lstm_parameters.pth')
    torch.save(e_pinn_model.lstm_model, '模型参数/lstm_.pt')
