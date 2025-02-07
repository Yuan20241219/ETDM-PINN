import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics.functional import r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

e_pinn_model=torch.load("E_pinn_model.pt")

if __name__ == '__main__':

    vali_data = pd.read_csv("../data/地震加速度.csv").iloc[202, 1:]
    v=e_pinn_model.predict(vali_data)

    result1 = -pd.read_csv("result_data/Newmark_beta_result.csv").iloc[0, 1:]
    etmd_result= pd.read_csv("result_data/ETDM_Newton_resutl.csv").iloc[:, 1]

    plt.figure(figsize=(15, 5))
    plt.plot(range(1500), v[0,0,0,:].detach(), label="PINN")
    plt.plot(range(1500), np.array(result1), label="NewMar", linestyle="--")
    plt.plot(range(1500), np.array(etmd_result), label="ETDM", linestyle="-.")
    plt.xlabel("时间步")
    plt.ylabel("位移/m")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(range(1500), v[0,0,0,:].detach()-np.array(result1), label="EPINN", linestyle="--")
    plt.plot(range(1500), np.array(etmd_result)-np.array(result1), label="ETDM", linestyle="-.")
    plt.xlabel("时间步")
    plt.ylabel("位移差值/m")
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("EPINN、ETDM局部迭代方法与NewMark方法计算值的差值")
    plt.show()

    r2 = r2_score(v[0,0,0,:], torch.from_numpy(np.array(result1)))
    print(r2)
