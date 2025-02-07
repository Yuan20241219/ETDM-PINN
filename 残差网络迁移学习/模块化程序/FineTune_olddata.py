import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from load_iter import SeqDataIter
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号



def V2Fnl(V):
    V_re = V[:, 0, 0, :]
    Fnl = 1000 * 1e6 * (V_re ** 3)
    return Fnl

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net=None, learning_rate=None):

    params_1x = [param for name, param in net.named_parameters()
         if name not in ["fc.weight", "fc.bias"]]
    trainer = torch.optim.Adam([{'params': params_1x},
                               {'params': net.fc.parameters(),
                                'lr': learning_rate }],
                            lr=learning_rate, weight_decay=0.998)

    return trainer

finetune_net = torch.load('模型参数/E_pinn_model.pt')
# finetune_net.lstm_model.fc = nn.Linear(finetune_net.lstm_model.fc.in_features, finetune_net.lstm_model.fc.out_features)
# nn.init.xavier_uniform_(finetune_net.lstm_model.fc.weight)
finetune_net.lstm_model.to("cuda")

finetune_net.optimizer=train_fine_tuning(finetune_net.lstm_model,1e-2)

Finetune_train=pd.read_csv("../../data/地震加速度.csv").iloc[182:212, 1:]
data=Finetune_train

# data_len=2500
# data=np.zeros((len(Finetune_train)//data_len,data_len))
# for i in range(len(Finetune_train)//data_len):
#     data[i,:]=Finetune_train.iloc[data_len*i:data_len*(i+1)].to_numpy()


loader=SeqDataIter(data,30,finetune_net.n_features,mean=finetune_net.mean,var=finetune_net.var)

timestamp1=time.time()
Finetune_loss=finetune_net.train(loader,50,[0], fun_V2F=V2Fnl,Fintune=True)
timestamp2=time.time()
time_difference = timestamp2 - timestamp1
print("时间差（秒）：", time_difference)

plt.figure(figsize=(15,5))
plt.plot(range(len(Finetune_loss)),Finetune_loss)
plt.show()

result1 = -pd.read_csv("../result_data/Newmark_beta_result.csv").iloc[0, 1:]
vali_data = pd.read_csv("../../data/地震加速度.csv").iloc[181, 1:] # m/s^2
v = finetune_net.predict(vali_data)

plt.figure(figsize=(15, 5))
plt.plot(range(len(v[0, 0, 0, :])), v[0, 0, 0, :].detach(), label="PINN")
plt.plot(range(len(result1)), np.array(result1), label="NewMar", linestyle="--")
plt.xlabel("时间步")
plt.ylabel("位移/m")
plt.grid(True)
plt.legend(loc='best')
plt.show()


before_result= pd.read_csv("../result_data/before_Fine.csv").iloc[:, 1]
print(before_result.shape)
plt.figure(figsize=(15, 5))
plt.plot(range(1500), v[0, 0, 0, :].detach() - np.array(result1), label="after", linestyle="--")
plt.plot(range(1500), np.array(before_result) - np.array(result1), label="before", linestyle="-.")
plt.xlabel("时间步")
plt.ylabel("位移差值/m")
plt.grid(True)
plt.legend(loc='best')
plt.title("EPINN、ETDM局部迭代方法与NewMark方法计算值的差值")
plt.show()





