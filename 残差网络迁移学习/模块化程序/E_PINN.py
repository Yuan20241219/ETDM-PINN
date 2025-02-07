import torch
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
from structure import Structure
from load_iter import SeqDataIter
from ETDM_matric import ETDM_structure


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(self.relu(out))
        out = self.relu(out)
        out = self.fc(out)
        return out


class SCL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_A):
        super(SCL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_A.shape[-2],kernel_A.shape[-1]), stride=1, bias=False)
        self.conv.weight.data = kernel_A
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        return x




class E_pinn():
    def __init__(self, n_features,A,lr):
        self.n_features = n_features
        self.lstm_model = LSTMModule(n_features, 60, 3, 1).to("cuda")
        self.A = A
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=lr,weight_decay=0.98)
        self.loss=nn.MSELoss()


    def train(self, load_iter, epochs,result_indx,V2Fnl):

        kern_A=self._get_kernA(self.A, result_indx)
        scl_model=SCL(1, len(result_indx), kern_A).to("cuda")
        train_loss = []
        for epoch in range(epochs):
            for batch_data in load_iter:
                self.optimizer.zero_grad()
                p = self.lstm_model(batch_data.to("cuda"))
                p_total = torch.cat((batch_data[:, :, 0].unsqueeze(2).to("cuda"), p), dim=2)
                P_total = torch.cat(
                    (torch.zeros([len(p_total), len(self.A) - 1, p_total.shape[2]]).to("cuda"), p_total), dim=1)
                P_scl = P_total.unsqueeze(1).permute(0, 1, 3, 2)
                V_nl = scl_model(P_scl)
                F_nl_re = V2Fnl(V_nl)
                V_loss = self.loss(p[:, :, 0],F_nl_re)
                V_loss.backward()
                self.optimizer.step()
            train_loss.append(V_loss)
        return torch.tensor(train_loss).to("cpu")

    def predict(self,data):
        kern_A=self.A.permute(1, 2, 0).unsqueeze(1)
        scl_model = SCL(1, self.A.shape[1], kern_A).to("cuda")
        batch_data=next(SeqDataIter(data,len(data),self.n_features))
        p = self.lstm_model(batch_data.to("cuda"))
        p_total = torch.cat((batch_data[:, :, 0].unsqueeze(2).to("cuda"), p), dim=2)
        P_total = torch.cat(
            (torch.zeros([len(p_total), len(self.A) - 1, p_total.shape[2]]).to("cuda"), p_total), dim=1)
        P_scl = P_total.unsqueeze(1).permute(0, 1, 3, 2)
        v = scl_model(P_scl)

        return v.to("cpu")

    def _get_kernA(self, A, result_indx):
        kern_A = A[:, result_indx, :]
        if len(kern_A.shape) == 2:
            kern_A = kern_A.unsqueeze(1)
        kern_A = kern_A.permute(1, 2, 0).unsqueeze(1)

        return kern_A

def V2Fnl(V):
    V_re = V[:, 0, 0, :]
    Fnl = 1000 * K[0] * (V_re ** 3)
    return Fnl

if __name__ == '__main__':
    dof = 5
    M = [5000] * 5
    K = [1e7] * 5
    zeta = 0.01
    dt = 0.02
    n_steps = 1500
    lr = 0.01
    yita = 1000
    epochs = 50

    train_data = pd.read_csv("../../data/地震加速度.csv").iloc[1:181, 1:]
    vali_data=pd.read_csv("../../data/地震加速度.csv").iloc[181, 1:]
    batch_size = 180
    n_features = 1

    result_indx = [0]

    L = torch.zeros((dof, 2))
    L[:, 0] = torch.tensor(M)
    L[:, 1] = -torch.tensor(([1, 0, 0, 0, 0]))

    structure = Structure(dof, M, K, num_omega1=1, num_omega2=5, zeta=zeta)
    structure = ETDM_structure(L, structure, dt=dt, n_steps=n_steps)
    load_iter = SeqDataIter(train_data, batch_size, n_features)


    e_pinn_model = E_pinn(n_features=n_features, A=structure.A, lr=lr)
    train_loss = e_pinn_model.train(load_iter, epochs, result_indx,V2Fnl)
    V=e_pinn_model.predict(vali_data)


    plt.figure(figsize=(15, 5))
    plt.plot(range(epochs), train_loss, label="训练损失")
    plt.xlabel("epoches")
    plt.ylabel("MESloss")
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(range(n_steps), V[0, 0, 0, :].to("cpu").detach(),label="PINN预测位移")
    plt.xlabel("epoches")
    plt.ylabel("MESloss")
    plt.legend(loc='best')
    plt.show()
