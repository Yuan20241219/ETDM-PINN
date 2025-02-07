import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from select import select


class Structure:

    def __init__(self, dof=0, M=None, K=None, C=None, *, num_omega1=None, num_omega2=None, zeta=None):
        """
        生成剪切变形为主的糖葫芦串模型的质量矩阵、刚度矩阵、阻尼矩阵、各阶频率、主阵型
        :param M: 质量列表，从下往上
        :param K: 层刚度列表，从下往上
        :param C:阻尼矩阵，当不填时可根据选择的频率阶数和阻尼比自动生成
        :param num_omega1: 当不填C时，用于生成阻尼矩阵的第一个自振频率阶数
        :param num_omega2: 当不填C时，用于生成阻尼矩阵的第二个自振频率阶数
        :param zeta: 当不填C时，计算阻尼矩阵的阻尼比
        """
        if dof==1:
            self.dof=1
            self.M=np.diag([M])
            self.K=np.diag([K])
            if C is None:
                self.C=2*zeta*pow(K*M,0.5)
            else:
                self.C=np.diag([C])
            self.omega=pow(K/M,0.5)
            self.vector=1

        else:

            self.dof = len(M)
            self.M = np.diag(M)
            self.K = self._get_K(K)
            self.omega ,self.vectors= self._get_omega(self.K, self.M)
            self.C = self._get_c(C,num_omega1,num_omega2,zeta)

    def _get_c(self, c, num1, num2, zeta):
        if c is None:
            a0 = 2 * zeta / (self.omega[num1-1] + self.omega[num2-1]) * (self.omega[num1-1] * self.omega[num2-1])
            a1 = 2 * zeta / (self.omega[num1-1] + self.omega[num2-1])
            C = a0 * self.M + a1 * self.K
            return C
        else:
            return c

    def _get_omega(self,M,K):
        eigeneigen, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(M), K))
        frequences = np.sqrt(eigeneigen)

        return np.flip(frequences),eigenvectors

    def get_info(self):
        print( self.__dict__)

    def _get_K(self, k):
        K=np.zeros((self.dof, self.dof))
        K[0,0]=k[0]+k[1]
        K[0,1]=-k[1]
        K[-1,-1]=k[-1]
        K[-1,-2]=-k[-1]
        for i in range(1,self.dof-1):
            K[i,i]=k[i]+k[i+1]
            K[i,i+1]=-k[i+1]
            K[i,i-1]=-k[i]
        return K


if __name__ == '__main__':
    M=5000
    K=1e6
    structure=Structure(dof=5,M=[M]*5,K=[K]*5,num_omega1=1,num_omega2=5,zeta=0.05)
    print(structure.get_info())

